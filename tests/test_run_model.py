"""Exercise the standalone driver without launching expensive MCFOST models."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_model


class RunModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.para = self.root / "chosen.para"
        self.para.write_bytes(b"supplied physical parameters\n")
        (self.root / "fidelity.txt").write_text(
            "eta: 3\nfidelity_stages:\n"
            "- name: sed_only\n  products: [sed]\n"
            "- name: joint\n  products: [sed, alma]\n")
        self.parameters = {"disk_pa": "17.5", "zone_1_Rc": "40", "nbr_photons_eq_th": "1e7"}
        self.pipeline = SimpleNamespace(
            ParaFile=Mock(return_value=SimpleNamespace(params=self.parameters)),
            run_mcfost=Mock(),
            load_and_score_outputs=Mock(return_value=(12.5, {"sed": {"chi2": 10}, "alma": {"loss": 2.5}})),
        )
        self.data = {"sed": [[1, 2], [3, 4], [0.1, 0.1]], "alma": {"image_spec": {"npix": 64}}}
        self.loader = Mock(return_value=self.data)
        self.pipeline_patch = patch.object(run_model, "load_pipeline", return_value=(self.loader, self.pipeline))
        self.pipeline_patch.start()
        self.addCleanup(self.pipeline_patch.stop)
        self.executable_patch = patch.object(run_model.shutil, "which", return_value="/fake/mcfost")
        self.executable_patch.start()
        self.addCleanup(self.executable_patch.stop)

    def args(self, *extra):
        return run_model.parse_args(["--working-root", str(self.root), *extra])

    def test_selected_model_and_observations_reach_shared_pipeline(self):
        original = self.para.read_bytes()
        report = run_model.evaluate(self.args("--correct-unresolved-polarimetry",
                                              "--unresolved-correction-radius-px", "3"))
        output = self.root / "model_results"
        self.assertEqual(self.para.read_bytes(), original)
        self.assertEqual((output / "model.para").read_bytes(), original)
        self.loader.assert_called_once_with("demo_ozstar", str(output), ["sed", "alma"])
        call = self.pipeline.run_mcfost.call_args
        self.assertEqual(call.args[1], output / "model.para")
        self.assertEqual(call.args[4]["disk_pa"], 17.5)
        self.assertEqual(call.args[4]["zone_1_Rc"], 40.0)
        self.assertEqual(call.args[4]["nbr_photons_eq_th"], "1e7")
        self.assertEqual(call.kwargs["alma_spec"], self.data["alma"]["image_spec"])
        scorer = self.pipeline.load_and_score_outputs.call_args
        self.assertTrue(scorer.args[3].correct_unresolved_polarimetry)
        self.assertEqual(scorer.args[3].unresolved_correction_radius_px, 3)
        self.assertEqual(report["cost"], 12.5)
        saved = json.loads((output / "evaluation.json").read_text())
        self.assertEqual(saved["additional_info"], report["additional_info"])
        self.assertEqual(saved["status"], "SUCCESS")

    def test_earlier_stage_loads_only_its_observations_and_copies_local_input(self):
        (self.root / "inputs").mkdir()
        (self.root / "inputs/waves.dat").write_text("1.0\n2.0\n")
        self.parameters["wavelength_file"] = "inputs/waves.dat"
        run_model.evaluate(self.args("--stage", "sed_only", "--puffed-up-rim", "0.1", "5", "2"))
        output = self.root / "model_results"
        self.loader.assert_called_once_with("demo_ozstar", str(output), ["sed"])
        self.assertEqual((output / "inputs/waves.dat").read_text(), "1.0\n2.0\n")
        call = self.pipeline.run_mcfost.call_args
        self.assertTrue(call.args[3])
        self.assertEqual(call.args[4]["puffed_r_rim"], 5)
        self.assertIsNone(call.kwargs["alma_spec"])

    def test_existing_output_is_not_reused(self):
        output = self.root / "model_results"
        output.mkdir()
        sentinel = output / "valuable_result"
        sentinel.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_model.evaluate(self.args())
        self.assertEqual(sentinel.read_text(), "keep me")
        self.pipeline.run_mcfost.assert_not_called()

    def test_pdi_profiles_use_the_supplied_parameter_file_grid(self):
        (self.root / 'fidelity.txt').write_text('fidelity_stages:\n- name: pdi\n  products: [pdi_V]\n')
        self.parameters.update(grid_nx='1000', grid_ny='1000', grid_size='800.', distance_pc='1220.')
        run_model.evaluate(self.args('--pdi-radial-bin-mas', '30'))
        self.loader.assert_called_once_with(
            'demo_ozstar', str(self.root / 'model_results'), ['pdi_V'],
            pdi_grid=self.parameters, pdi_radial_bin_mas=30.)

    def test_ambiguous_para_and_unknown_stage_fail_before_execution(self):
        (self.root / "another.para").write_text("different model")
        with self.assertRaisesRegex(ValueError, "--para-file"):
            run_model.evaluate(self.args())
        with self.assertRaisesRegex(ValueError, "Unknown stage"):
            run_model.evaluate(self.args("--para-file", "chosen.para", "--stage", "missing"))
        self.pipeline.run_mcfost.assert_not_called()

    def test_empty_sed_aborts_before_simulation(self):
        self.data["sed"] = [[], [], []]
        with self.assertRaisesRegex(ValueError, "SED observations are empty"):
            run_model.evaluate(self.args())
        self.pipeline.run_mcfost.assert_not_called()
        saved = json.loads((self.root / "model_results/evaluation.json").read_text())
        self.assertEqual(saved["status"], "FAILED")
        self.assertNotIn("cost", saved)

    def test_failure_sentinel_is_not_reported_as_measured_cost(self):
        self.pipeline.load_and_score_outputs.return_value = (1e99, {"failure": {"code": "missing_sed_output"}})
        with self.assertRaisesRegex(RuntimeError, "Model comparison failed"):
            run_model.evaluate(self.args())
        saved = json.loads((self.root / "model_results/evaluation.json").read_text())
        self.assertEqual(saved["status"], "FAILED")
        self.assertEqual(saved["additional_info"]["failure"]["code"], "missing_sed_output")
        self.assertNotIn("cost", saved)

    def test_correction_requires_radius(self):
        with self.assertRaises(SystemExit):
            self.args("--correct-unresolved-polarimetry")


if __name__ == "__main__":
    unittest.main()
