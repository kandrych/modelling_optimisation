"""Compare trial and final MCFOST inputs without launching simulations."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import SMAC3


class TrialConfig(dict):
    config_id = 7


class PuffedRimTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = self.root / "smac_results"
        self.args = SimpleNamespace(puffed_up_rim=True, plot_intermediate=False,
                                    two_zone_cont_lhc=False, tapered_edge_p1_eq_p2=False,
                                    share_zone_composition=False, fidelity_settings={})
        self.fidelity = {"stage": "sed", "products": ["sed"]}
        # Minimal one-zone template, following the real parser's line layout.
        lines = ["1 1 1 1 1 1 1\n"] * 41
        lines[39:41] = ["#Number of zones\n", "1\n"]
        lines += ["#Density structure\n", "1\n", "0.004 100\n", "1 10 2\n",
                  "10 0 100 40\n", "1.1\n", "-1 -0.3\n", "\n",
                  "#Grain properties\n", "1\n", "Mie 1 1 0 1 0.7\n",
                  "silicate.lnk 1\n", "1\n", "0.1 1000 3.5 100\n",
                  "#Star properties\n", "1\n", "6000 1 1 0 0 0 T\n", "star.fits\n", "0 0\n"]
        (self.root / "simulation.para").write_text("".join(lines))

    def write_final_parameters(self, incumbent):
        # Exercise the actual final-input call in main(), without optimisation.
        tree = ast.parse(Path(SMAC3.__file__).read_text())
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        call = next(node for node in main.body if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "write_mcfost_paramfile")
        namespace = dict(vars(SMAC3), incumbent=incumbent, fidelity_result=self.fidelity,
                         results_dir=self.results, args=self.args)
        exec(compile(ast.Module(body=[call], type_ignores=[]), "<final input>", "exec"), namespace)
        return namespace["par_path"]

    def compare_trial_and_final(self, values):
        config = TrialConfig(values)
        trial_dir = self.root / "trials" / "config_7"
        config._obriy_trial_dir = str(trial_dir)
        incumbent = dict(values)
        with patch.object(SMAC3, "map_budget_to_fidelity", return_value=self.fidelity), \
             patch.object(SMAC3.obm, "run_mcfost_safe") as command, \
             patch.object(SMAC3.obm, "load_and_score_outputs", side_effect=lambda *a: (12.0, {})) as score, \
             redirect_stdout(io.StringIO()):
            SMAC3.objective(config, 1, 1.0, {}, str(self.root / "trials"), self.args)
            trial_commands = [call.kwargs for call in command.call_args_list]
            trial_scoring_cfg = score.call_args.args[-1]
            command.reset_mock()
            par_path = self.write_final_parameters(incumbent)
            SMAC3.final_model(config, 1, 1.0, fidelity_result=self.fidelity,
                              par_path=par_path, results_dir=self.results,
                              data_arg={}, args=self.args, incumbent=incumbent)
            self.assertEqual(trial_commands, [call.kwargs for call in command.call_args_list])
            self.assertEqual(trial_scoring_cfg, score.call_args.args[-1])
        self.assertEqual(dict(config), values)
        self.assertEqual(incumbent, values)
        self.assertEqual((trial_dir / "model.para").read_text(),
                         (self.results / "model.para").read_text())
        trial_record = json.loads((trial_dir / "config_used.json").read_text())
        final_record = json.loads((self.results / "config_used.json").read_text())
        self.assertEqual(trial_record, final_record)
        return trial_commands, final_record["cfg"]

    def test_rim_radius_and_all_command_options_match(self):
        commands, recorded = self.compare_trial_and_final({
            "zone_1_Rin": "7.0", "puffed_r_offset": "0.5",
            "puffed_h_rim_over_h0": 1.2, "puffed_delta_r": 0.1})
        self.assertEqual(commands, [
            {"options": ["-puffed_up_rim", "1.2", "7.5", "0.1"], "logfile": "mcfost_temp.log"},
            {"options": ["-disk_struct", "-puffed_up_rim", "1.2", "7.5", "0.1"],
             "logfile": "mcfost_base_struct.log"},
        ])
        self.assertEqual(recorded["puffed_r_rim"], 7.5)

    def test_disabled_rim_does_not_require_or_add_rim_parameters(self):
        self.args.puffed_up_rim = False
        commands, recorded = self.compare_trial_and_final({"zone_1_Rin": 7.0})
        self.assertNotIn("puffed_r_rim", recorded)
        self.assertEqual(commands, [{"logfile": "mcfost_temp.log"},
                                   {"options": ["-disk_struct"], "logfile": "mcfost_base_struct.log"}])

    def test_missing_offset_is_rejected_in_both_paths_before_simulation(self):
        config = TrialConfig(zone_1_Rin=7.0)
        config._obriy_trial_dir = str(self.root / "trials" / "config_7")
        with patch.object(SMAC3, "map_budget_to_fidelity", return_value=self.fidelity), \
             patch.object(SMAC3.obm, "run_mcfost_safe") as command, \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(KeyError, "puffed_r_offset"):
                SMAC3.objective(config, 1, 1.0, {}, str(self.root / "trials"), self.args)
            with self.assertRaisesRegex(KeyError, "puffed_r_offset"):
                self.write_final_parameters(dict(config))
            with self.assertRaisesRegex(KeyError, "puffed_r_offset"):
                SMAC3.final_model(config, 1, 1.0, fidelity_result=self.fidelity,
                                  par_path=self.results / "model.para", results_dir=self.results,
                                  data_arg={}, args=self.args, incumbent=dict(config))
            command.assert_not_called()

    def test_all_image_products_keep_image_options_with_rim_enabled_or_disabled(self):
        products = ('vis2_1perband', 'vis2_chromatic', 'pdi_V', 'pdi_I', 'pdi_H', 'alma')
        cfg = dict(puffed_h_rim_over_h0=1.2, puffed_r_rim=7.5, puffed_delta_r=0.1)
        rim_options = ['-puffed_up_rim', '1.2', '7.5', '0.1']
        for enabled in (False, True):
            for product in products:
                with self.subTest(enabled=enabled, product=product), \
                        patch.object(SMAC3.obm, 'run_mcfost_safe') as command, \
                        redirect_stdout(io.StringIO()):
                    workdir = self.root / f'{product}_{enabled}'
                    workdir.mkdir()
                    SMAC3.obm.run_mcfost(
                        {'products': [product]}, self.root / 'simulation.para', workdir,
                        enabled, cfg if enabled else {},
                        alma_spec={'npix': 64, 'fov_mas': 500., 'wavelength_um': 870.})
                    image_calls = command.call_args_list[2:]
                    self.assertTrue(image_calls)
                    for call in image_calls:
                        options = call.kwargs['options']
                        self.assertEqual(options[0], '-img')
                        self.assertGreater(float(options[1]), 0)
                        image_options = options[:2]
                        if product == 'alma':
                            self.assertEqual(options[1], '870.000')
                            image_options += ['-casa']
                        self.assertEqual(options, image_options + (rim_options if enabled else []))


if __name__ == "__main__":
    unittest.main()
