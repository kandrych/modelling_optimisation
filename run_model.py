#!/usr/bin/env python3
"""Run a supplied MCFOST .para file and score it with the SMAC3 pipeline."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number.")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-root", type=Path, required=True,
                        help="Folder containing the .para file and fidelity.txt.")
    parser.add_argument("--para-file", type=Path,
                        help="Parameter file, relative to working root or absolute; required if ambiguous.")
    parser.add_argument("--fidelity-config", type=Path, default=Path("fidelity.txt"),
                        help="Fidelity file, relative to working root or absolute.")
    parser.add_argument("--stage", help="Stage name; defaults to the last stage in the fidelity file.")
    parser.add_argument("--output-dir", type=Path, default=Path("model_results"),
                        help="New output directory, relative to working root or absolute.")
    parser.add_argument("--data-root", default="demo_ozstar",
                        choices=("demo_ozstar", "demo_mac", "ar_pup_ozstar"),
                        help="Observation preset used by SMAC3.load_data (default: demo_ozstar).")
    parser.add_argument("--plot-intermediate", action=argparse.BooleanOptionalAction, default=True,
                        help="Save model comparison plots (enabled by default).")
    parser.add_argument("--correct-unresolved-polarimetry", action="store_true")
    parser.add_argument("--unresolved-correction-radius-px", type=positive_float)
    parser.add_argument("--pdi-radial-bin-mas", type=positive_float, default=25.0)
    parser.add_argument("--overresolved_flux_fit_for_interferometry", type=positive_float,
                        help="Optional background-fit reference wavelength in microns, as in SMAC3.py.")
    parser.add_argument("--puffed-up-rim", type=float, nargs=3,
                        metavar=("H_OVER_R", "R_AU", "DELTA_R"),
                        help="Optional MCFOST puffed-rim arguments; these are not stored in .para.")
    args = parser.parse_args(argv)
    if args.correct_unresolved_polarimetry and args.unresolved_correction_radius_px is None:
        parser.error("--correct-unresolved-polarimetry requires --unresolved-correction-radius-px.")
    if args.puffed_up_rim is not None and not all(math.isfinite(x) for x in args.puffed_up_rim):
        parser.error("Puffed-rim arguments must be finite.")
    return args


def resolve_inputs(args):
    root = args.working_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Working root is not a directory: {root}")

    def resolve(path):
        return (root / path.expanduser()).resolve()

    if args.para_file is None:
        candidates = sorted(root.glob("*.para"))
        if len(candidates) != 1:
            raise ValueError("Specify --para-file: expected exactly one .para file directly in working root; "
                             f"found {len(candidates)}.")
        para = candidates[0].resolve()
    else:
        para = resolve(args.para_file)
    fidelity = resolve(args.fidelity_config)
    for path in (para, fidelity):
        if not path.is_file():
            raise ValueError(f"Input file does not exist: {path}")
    output = resolve(args.output_dir)
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}. Choose a new --output-dir.")
    return para, fidelity, output


def select_fidelity(settings, stage_name):
    if stage_name is None:
        return settings["stages"][-1]
    for stage in settings["stages"]:
        if stage["stage"] == stage_name:
            return stage
    names = ", ".join(stage["stage"] for stage in settings["stages"])
    raise ValueError(f"Unknown stage {stage_name!r}; choose from: {names}")


def local_input_files(parameters, source_dir):
    """Preserve local wavelength, dust and stellar files when changing run directory.

    Missing local files are left to MCFOST's normal utility-file lookup. Absolute
    references need no copying. Parent-relative local references cannot be copied
    safely into an isolated output directory without changing the .para file.
    """
    files = set()
    for key, value in parameters.items():
        if key != "wavelength_file" and not key.endswith(("_spectrum_file", "_optical_indices_file")):
            continue
        path = Path(str(value))
        if path.is_absolute() or not (source_dir / path).is_file():
            continue
        if ".." in path.parts:
            raise ValueError(f"Local reference {value!r} in {key} contains '..'. "
                             "Use an absolute reference in your input .para file.")
        if path.parts[0] in {"model.para", "fidelity.txt", "evaluation.json", "data_th", "figures", "data_figures"}:
            raise ValueError(f"Local reference conflicts with an output path: {path}")
        files.add(path)
    return sorted(files)


def json_ready(value):
    """Keep NumPy results readable as standard JSON; unmeasured values become null."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_report(output, report):
    temporary = output / "evaluation.json.tmp"
    temporary.write_text(json.dumps(json_ready(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "evaluation.json")


def load_pipeline():
    # Importing SMAC3 defines its functions; it does not start optimisation.
    os.environ["MPLBACKEND"] = "Agg"
    from SMAC3 import load_data
    from lib import obriy_mcfost
    return load_data, obriy_mcfost


def evaluate(args):
    from lib.obriy_fidelity import load_fidelity_config

    para, fidelity_path, output = resolve_inputs(args)
    settings = load_fidelity_config(fidelity_path)
    fidelity = select_fidelity(settings, args.stage)
    load_data, mcfost = load_pipeline()
    parameters = mcfost.ParaFile(str(para)).params
    cfg = dict(parameters)
    # ParaFile returns strings; these two values feed numerical plot/scoring code.
    for key in ("disk_pa", "zone_1_Rc"):
        if key in cfg:
            cfg[key] = float(cfg[key])
    if args.puffed_up_rim is not None:
        cfg.update(zip(("puffed_h_rim_over_h0", "puffed_r_rim", "puffed_delta_r"), args.puffed_up_rim))
    auxiliary_files = local_input_files(parameters, para.parent)
    if shutil.which("mcfost") is None:
        raise RuntimeError("mcfost is not available in PATH. Load your normal modelling environment.")

    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING", "source_para": str(para), "source_fidelity": str(fidelity_path),
        "output_dir": str(output), "fidelity": fidelity, "arguments": vars(args),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "auxiliary_files": auxiliary_files,
    }
    started = time.monotonic()
    try:
        write_report(output, report)
        model_path = output / "model.para"
        shutil.copy2(para, model_path)
        shutil.copy2(fidelity_path, output / "fidelity.txt")
        for relative in auxiliary_files:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(para.parent / relative, destination)
        print(f"[run_model] Stage: {fidelity['stage']}; products: {fidelity['products']}", flush=True)
        print(f"[run_model] Output: {output}", flush=True)
        data = load_data(args.data_root, str(output), fidelity["products"])
        # The shared loader can return empty SED arrays after a read failure.
        # Refuse to turn missing observations into a zero contribution.
        if "sed" in fidelity["products"] and any(len(column) == 0 for column in data["sed"]):
            raise ValueError("SED observations are empty; check the selected observation preset and files.")
        mcfost.run_mcfost(
            fidelity, model_path, output, args.puffed_up_rim is not None, cfg,
            alma_spec=data["alma"]["image_spec"] if "alma" in fidelity["products"] else None,
        )
        loss, additional_info = mcfost.load_and_score_outputs(fidelity, output, data, args, cfg)
        report["additional_info"] = additional_info
        if "failure" in additional_info:
            raise RuntimeError(f"Model comparison failed: {additional_info['failure']}")
        if not math.isfinite(float(loss)):
            raise RuntimeError(f"Model comparison returned a non-finite cost: {loss}")
        report.update(status="SUCCESS", cost=float(loss))
    except Exception as error:
        report.update(status="FAILED", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(output, report)
    print(f"[run_model] Total cost: {float(loss):.12g}", flush=True)
    print(f"[run_model] Scores: {output / 'evaluation.json'}", flush=True)
    return report


def main(argv=None):
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
