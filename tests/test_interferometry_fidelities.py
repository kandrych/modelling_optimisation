"""Exercise the real loading/scoring routing without MCFOST or SMAC imports."""
import __future__
import ast
from contextlib import nullcontext, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from lib.obriy_interferometry_data import load_interferometry_data
from lib.obriy_fidelity import load_fidelity_config, fidelity_products


ROOT = Path(__file__).resolve().parents[1]
MODES = ("vis2_1perband", "vis2_chromatic")
INSTRUMENTS = ("pionier", "gravity", "matisse_l", "matisse_n")


def extract(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def fake_reader(directory, pattern, **kwargs):
    return SimpleNamespace(directory=directory, pattern=pattern, selection=kwargs,
                           vis_in_fcorr=kwargs.get("fcorr", False))


class InterferometryFidelityTests(unittest.TestCase):
    def test_existing_selections_and_single_modes(self):
        for root in ("demo_mac", "demo_ozstar"):
            both = load_interferometry_data(root, MODES, fake_reader)
            for mode in MODES:
                alone = load_interferometry_data(root, [mode], fake_reader)
                self.assertEqual(list(alone), [mode])
                for instrument in INSTRUMENTS:
                    self.assertEqual(vars(alone[mode][instrument]), vars(both[mode][instrument]))
            self.assertEqual(both[MODES[0]]["pionier"].selection, {"wave_lims": (1.63, 1.64)})
            self.assertEqual(both[MODES[0]]["gravity"].selection, {"wave_lims": (2.199, 2.201)})
            self.assertEqual(both[MODES[0]]["matisse_l"].selection, {"wave_lims": (3.48, 3.52)})
            self.assertEqual(both[MODES[0]]["matisse_n"].selection,
                             {"wave_lims": (9.9, 10.10), "fcorr": True})
            self.assertEqual(both[MODES[1]]["pionier"].selection, {})
            self.assertEqual(both[MODES[1]]["gravity"].selection, {})
            self.assertEqual(both[MODES[1]]["matisse_l"].selection,
                             {"wave_lims": (2.95, 3.95), "v2lim": 1e-8})
            self.assertEqual(both[MODES[1]]["matisse_n"].selection,
                             {"wave_lims": (8.0, 13.0), "v2lim": 1e-8, "fcorr": True})

    def test_no_interferometry_and_failed_reads(self):
        reader = Mock(side_effect=OSError("missing FITS"))
        self.assertEqual(load_interferometry_data("ar_pup_ozstar", ["sed"], reader), {})
        reader.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "vis2_chromatic.*pionier"):
            load_interferometry_data("demo_ozstar", [MODES[1]], reader)

    def test_real_loader_and_stage_scoring(self):
        for root in ("demo_mac", "demo_ozstar"):
            with self.subTest(root=root), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                work = Path(directory)
                config = work / "fidelity.txt"
                config.write_text("fidelity_stages:\n"
                                  "- name: sed\n  products: [sed]\n"
                                  "- name: narrow\n  products: [sed, vis2_1perband]\n"
                                  "- name: chromatic\n  products: [sed, vis2_chromatic]\n")
                settings = load_fidelity_config(config)
                validator = Mock()
                loader = extract("SMAC3.py", "load_data", dict(
                    Path=Path, load_interferometry_data=load_interferometry_data,
                    distroi=SimpleNamespace(read_oi_container_from_oifits=fake_reader),
                    obi=SimpleNamespace(validate_interferometric_data=validator),
                    obs=SimpleNamespace(load_sed_data=lambda path: ([1, 2], [3, 4], [0.1, 0.1]))))
                data = loader(root, directory, fidelity_products(settings))
                self.assertEqual(validator.call_count, 8)
                self.assertEqual(set(data["interferometry"]), set(MODES))
                self.assertIn("image_spec", data["alma"])
                self.assertIn("pol_images", data["pdi_V"])
                (work / "data_th").mkdir()
                (work / "data_th/sed_rt.fits.gz").touch()
                for stage in settings["stages"]:
                    for background in (None, 2.2):
                        obi = SimpleNamespace(
                            monochromatic_chi=Mock(return_value=(2., 1., -1., 3)),
                            chromatic_chi=Mock(return_value=(2., 1., -1., 3)),
                            monochromatic_chi_with_background=Mock(return_value=(2., 1., -1., 3, 0.1)))
                        scorer = extract("lib/obriy_mcfost.py", "load_and_score_outputs", dict(
                            os=os, obi=obi,
                            obg=SimpleNamespace(diagnostic_plot=lambda *args: nullcontext()),
                            plt=SimpleNamespace(get_fignums=lambda: []),
                            plot_mcfost_disk_structure=Mock(), plot_mcfost_density_temperature_cuts=Mock(),
                            distroi=SimpleNamespace(read_sed_mcfost=Mock()),
                            obs=SimpleNamespace(chi2_SED_with_reddening=Mock(return_value=(4., 2., -2., 0.)))))
                        loss, info = scorer(stage, work, data, SimpleNamespace(
                            plot_intermediate=True, overresolved_flux_fit_for_interferometry=background), {})
                        mode = next((m for m in MODES if m in stage["products"]), None)
                        self.assertEqual(loss, 2. if mode is None else 6.)
                        if mode is None:
                            obi.monochromatic_chi.assert_not_called()
                            obi.chromatic_chi.assert_not_called()
                            continue
                        method = (obi.chromatic_chi if mode == MODES[1] else
                                  obi.monochromatic_chi_with_background if background else obi.monochromatic_chi)
                        calls = method.call_args_list[-4:]
                        self.assertEqual(len(calls), 4)
                        for instrument, call in zip(INSTRUMENTS, calls):
                            self.assertIs(call.kwargs["container_data"], data["interferometry"][mode][instrument])
                            self.assertTrue(call.kwargs["plot"])
                            self.assertEqual(info[mode][instrument]["num_points"], 3)
                        if background:
                            reference = obi.monochromatic_chi_with_background.call_args_list[0]
                            self.assertIs(reference.kwargs["container_data"], data["interferometry"][mode]["gravity"])

    def test_chromatic_score_and_plot_share_observations(self):
        data = fake_reader("PIONIER", "*.fits")
        model = object()
        score, plot = Mock(return_value=(2., 1., -1., 3)), Mock()
        function = extract("lib/obriy_interferometry.py", "chromatic_chi", dict(
            Path=Path, glob=SimpleNamespace(glob=lambda *a, **k: ["image"]),
            distroi=SimpleNamespace(read_image_mcfost=lambda path: SimpleNamespace(wavelength=1.65)),
            calc_observables_with_secondary=Mock(return_value=model), oi_container_chi2=score,
            oi_container_plot_data_vs_model=plot, plot_secondary_comparison=Mock(),
            obg=SimpleNamespace(diagnostic_plot=lambda *args: nullcontext())))
        function("trial", ["data_1.65"], data, img_sed=object(), plot=True)
        self.assertIs(score.call_args.args[0], data)
        self.assertIs(plot.call_args.args[0], data)
        self.assertIs(plot.call_args.args[1], model)


if __name__ == "__main__":
    unittest.main()
