#!/usr/bin/env python3
"""Score existing MCFOST outputs and optionally regenerate their comparison plots."""
import argparse
import json
import math
from pathlib import Path
import time

from run_model import json_ready, load_pipeline, positive_float, select_fidelity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Existing folder containing model.para and MCFOST data_* outputs.")
    parser.add_argument("--fidelity-config", type=Path, required=True,
                        help="Fidelity file; relative paths are resolved from the current directory.")
    parser.add_argument("--stage", help="Stage name; defaults to the last stage in the fidelity file.")
    parser.add_argument("--data-root", default="demo_ozstar",
                        choices=("demo_ozstar", "demo_mac", "ar_pup_ozstar"),
                        help="Observation preset used by SMAC3.load_data (default: demo_ozstar).")
    parser.add_argument("--plot-intermediate", action=argparse.BooleanOptionalAction, default=True,
                        help="Regenerate model comparison plots (enabled by default).")
    parser.add_argument("--correct-unresolved-polarimetry", action="store_true")
    parser.add_argument("--unresolved-correction-radius-px", type=positive_float)
    parser.add_argument("--pdi-radial-bin-mas", type=positive_float, default=25.0)
    parser.add_argument("--overresolved_flux_fit_for_interferometry", type=positive_float,
                        help="Optional background-fit reference wavelength in microns, as in SMAC3.py.")
    args = parser.parse_args(argv)
    if args.correct_unresolved_polarimetry and args.unresolved_correction_radius_px is None:
        parser.error("--correct-unresolved-polarimetry requires --unresolved-correction-radius-px.")

    model_dir = args.model_dir.expanduser().resolve()
    para_path = model_dir / "model.para"
    fidelity_path = args.fidelity_config.expanduser().resolve()
    for path in (para_path, fidelity_path):
        if not path.is_file():
            raise ValueError(f"Input file does not exist: {path}")

    from lib.obriy_fidelity import load_fidelity_config

    settings = load_fidelity_config(fidelity_path)
    fidelity = select_fidelity(settings, args.stage)
    # Use the parameters that actually produced these outputs, including template defaults.
    load_data, mcfost = load_pipeline()
    parameters = mcfost.ParaFile(str(para_path)).params
    cfg = dict(parameters)
    # ParaFile returns strings; these values also feed numerical diagnostics.
    for key in ("disk_pa", "zone_1_Rc"):
        if key in cfg:
            cfg[key] = float(cfg[key])

    report = {
        "status": "RUNNING", "model_dir": model_dir, "source_para": para_path,
        "source_fidelity": fidelity_path, "fidelity": fidelity, "arguments": vars(args),
    }
    started = time.monotonic()
    try:
        print(f"[score_model] Model: {model_dir}", flush=True)
        print(f"[score_model] Stage: {fidelity['stage']}; products: {fidelity['products']}", flush=True)
        # Load only the selected observations. PDI profiles use this model's fixed grid.
        if any(product.startswith("pdi_") for product in fidelity["products"]):
            data = load_data(args.data_root, str(model_dir), fidelity["products"],
                             pdi_grid=parameters, pdi_radial_bin_mas=args.pdi_radial_bin_mas)
        else:
            data = load_data(args.data_root, str(model_dir), fidelity["products"])

        # MCFOST has already finished: read its outputs directly through the shared scorer.
        loss, additional_info = mcfost.load_and_score_outputs(fidelity, model_dir, data, args, cfg)
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
        # Keep the original run's evaluation.json. This report describes the new scoring only.
        temporary = model_dir / "scoring.json.tmp"
        temporary.write_text(json.dumps(json_ready(report), indent=2, allow_nan=False) + "\n")
        temporary.replace(model_dir / "scoring.json")

    print(f"[score_model] Total cost: {float(loss):.12g}", flush=True)
    print(f"[score_model] Scores: {model_dir / 'scoring.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
