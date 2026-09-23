"""Dependency-light checks of mass splitting and parameter-writer integration.

Extract the relevant functions to avoid importing MCFOST plotting dependencies
or starting the SMAC application during these tests.
"""
import ast
import __future__
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.integrate import quad


ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT / "lib/obriy_mcfost.py").read_text())
names = {"_2zone_cont_calc_dmass", "write_mcfost_paramfile",
         "_apply_selected_second_component_fractions"}
namespace = dict(np=np, json=json, Path=Path, quad=quad)
exec(compile(ast.Module(body=[n for n in source.body
                             if isinstance(n, ast.FunctionDef) and n.name in names],
                        type_ignores=[]), str(ROOT / "lib/obriy_mcfost.py"), "exec",
             flags=__future__.annotations.compiler_flag), namespace)


class TwoZoneTests(unittest.TestCase):
    def test_real_parameter_roundtrip_and_smac_calls(self):
        # Use the real parser/writer with a synthetic template in its expected
        # layout. This is a file-I/O test, not an MCFOST simulation fixture.
        real_ns = namespace.copy()
        parser_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ParaFile')
        nodes = [parser_class] + [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<real writer>', 'exec',
                     flags=__future__.annotations.compiler_flag), real_ns)
        smac_tree = ast.parse((ROOT/'SMAC3.py').read_text())
        functions = [n for n in smac_tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('objective', 'make_unique_config_trial_dir')]
        main = next(n for n in smac_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        final_write = next(n for n in main.body if isinstance(n, ast.Assign)
                           and isinstance(n.value, ast.Call)
                           and isinstance(n.value.func, ast.Attribute)
                           and n.value.func.attr == 'write_mcfost_paramfile')
        arg_nodes = [n for n in ast.walk(main) if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and n.value.args
                     and isinstance(n.value.args[0], ast.Constant)
                     and n.value.args[0].value in ('--2ZONE_CONT_LHC', '--tapered-edge-p1-eq-p2', '--share-zone-composition')]
        arg_parser = argparse.ArgumentParser()
        exec(compile(ast.Module(body=arg_nodes, type_ignores=[]), '<flags>', 'exec'), {'p': arg_parser})

        class Config(dict):
            config_id = 1

        for count, outer_type in [(1, 2), (2, 1), (2, 2)]:
            for continuous in (False, True):
                if count == 1 and continuous:
                    continue
                for tied in (False, True):
                    with self.subTest(zones=count, outer_type=outer_type, continuous=continuous, tied=tied):
                        flags = (['--2ZONE_CONT_LHC'] if continuous else []) + (['--tapered-edge-p1-eq-p2'] if tied else [])
                        args = arg_parser.parse_args(flags)
                        args.puffed_up_rim = False
                        # The parser maps the header by fixed line indices.
                        lines = ['1 1 1 1 1 1 1\n'] * 41
                        lines[39] = '#Number of zones\n'
                        lines[40] = f'{count}\n'
                        lines.append('#Density structure\n')
                        for zone in range(1, count+1):
                            kind = outer_type if zone == count else 1
                            lines.extend([f'{kind}\n', '0.004 100\n', '1 10 2\n',
                                          '10 0 100 40\n', '1.1\n', '-1 -0.3\n', '\n'])
                        lines.append('#Grain properties\n')
                        for zone in range(count):
                            lines.extend([f'{zone+1}\n', 'Mie 1 1 0 1 0.7\n', 'silicate.lnk 1\n',
                                          '1\n', '0.1 1000 3.5 100\n'])
                            if zone == 1:
                                lines.extend(['Mie 2 1 0 0.1 0.7\n', 'carbon.lnk 0.8\n',
                                              'ice.lnk 0.2\n', '3\n', '0.001 1 3.5 10\n'])
                        lines.extend(['#Star properties\n', '1\n', '6000 1 1 0 0 0 T\n',
                                      'star.fits\n', '0 0\n'])
                        with tempfile.TemporaryDirectory() as directory:
                            root = Path(directory)
                            (root/'simulation.para').write_text(''.join(lines))
                            (root/'trials').mkdir()
                            cfg = Config({f'zone_{count}_surface_density_exp': -1.5})
                            if continuous:
                                cfg.update(disk_Rmid=20, disk_total_dust_mass=0.009)
                            obm = SimpleNamespace(write_mcfost_paramfile=real_ns['write_mcfost_paramfile'],
                                                  run_mcfost=Mock(),
                                                  load_and_score_outputs=Mock(return_value=(12.0, {'sed': {}})))
                            ns = dict(Path=Path, obm=obm, map_budget_to_fidelity=lambda b: {'products': ['sed']})
                            exec(compile(ast.Module(body=functions, type_ignores=[]), '<objective>', 'exec',
                                         flags=__future__.annotations.compiler_flag), ns)
                            self.assertEqual(ns['objective'](cfg, 1, 0.1, [], str(root/'trials'), args), (12.0, {'sed': {}}))
                            obm.run_mcfost.assert_called_once()
                            trial_path = obm.run_mcfost.call_args.args[1]
                            ns.update(incumbent=dict(cfg), fidelity_result={'products': ['sed']},
                                      results_dir=root/'smac_results', args=args)
                            exec(compile(ast.Module(body=[final_write], type_ignores=[]), '<final write>', 'exec'), ns)
                            trial = real_ns['ParaFile'](trial_path).params
                            final = real_ns['ParaFile'](ns['par_path']).params
                            self.assertEqual(trial, final)
                            self.assertEqual(float(trial[f'zone_{count}_-gamma_exp']), -1.5 if tied else -0.3)
                            if count == 2:
                                self.assertEqual(float(trial['zone_1_-gamma_exp']), -0.3)
                            if continuous:
                                self.assertEqual(float(trial['zone_1_Rout']), 20)
                                self.assertEqual(float(trial['zone_2_Rin']), 20)
                                m1, m2 = [float(trial[f'zone_{z}_dust_mass']) for z in (1, 2)]
                                self.assertAlmostEqual(m1+m2, 0.009)
                                s1, s2 = [float(trial[f'zone_{z}_surface_density_exp']) for z in (1, 2)]
                                q = 2 + float(trial['zone_2_-gamma_exp'])
                                profile = lambda r: r**s2 * (np.exp(-(r/40)**q) if outer_type == 2 else 1)
                                sigma1 = m1*20**s1 / quad(lambda r:r**(s1+1), 10, 20)[0]
                                sigma2 = m2*profile(20) / quad(lambda r:r*profile(r), 20, 100)[0]
                                np.testing.assert_allclose(sigma1, sigma2, rtol=1e-9, atol=0)
                            else:
                                self.assertEqual(float(trial['zone_1_dust_mass']), 0.004)
                            if count == 2:
                                self.assertEqual(trial['zone_2_number_of_species'], '2')
                                args.share_zone_composition = arg_parser.parse_args(
                                    ['--share-zone-composition']).share_zone_composition
                                cfg['zone_1_species_1_amin'] = 0.2
                                cfg['zone_1_species_1_component_1_optical_indices_file'] = 'new_material.lnk'
                                ns['objective'](cfg, 1, 0.1, [], str(root/'trials'), args)
                                shared_path = obm.run_mcfost.call_args.args[1]
                                shared = real_ns['ParaFile'](shared_path).params
                                ns['incumbent'] = dict(cfg)
                                exec(compile(ast.Module(body=[final_write], type_ignores=[]), '<shared final>', 'exec'), ns)
                                self.assertEqual(shared, real_ns['ParaFile'](ns['par_path']).params)
                                recorded = json.loads((shared_path.parent/'config_used.json').read_text())['cfg']
                                for key in shared:
                                    if key.startswith('zone_1_species_') or key == 'zone_1_number_of_species':
                                        target = key.replace('zone_1_', 'zone_2_', 1)
                                        self.assertEqual(shared[key], shared[target])
                                        self.assertEqual(str(recorded[target]), shared[target])
                                    elif not key.startswith('zone_2_species_') and key != 'zone_2_number_of_species':
                                        self.assertEqual(shared[key], trial[key])
                                self.assertNotIn('zone_2_species_2_grain_type', shared)
                                self.assertNotIn('zone_2_species_1_amin', cfg)
                                with self.assertRaisesRegex(ValueError, 'remove from config'):
                                    real_ns['write_mcfost_paramfile'](
                                        dict(cfg, zone_2_species_1_amin=0.5), {}, root/'invalid_shared',
                                        share_zone_composition=True)
                            else:
                                with self.assertRaisesRegex(ValueError, 'exactly two'):
                                    real_ns['write_mcfost_paramfile'](
                                        cfg, {}, root/'invalid_shared', share_zone_composition=True)

    def config(self, inner=-1.0, outer=-1.0):
        return dict(disk_Rmid=20.0, disk_total_dust_mass=0.009,
                    zone_1_Rin=10.0, zone_2_Rout=100.0,
                    zone_1_surface_density_exp=inner,
                    zone_2_surface_density_exp=outer)

    def test_mass_and_boundary_density(self):
        for inner, outer in [(-1, -1), (-2, -2), (-2, 0.5),
                             (0.5, -2), (-2 + 1e-10, -2 - 1e-10), (-3, 1)]:
            with self.subTest(inner=inner, outer=outer):
                cfg = self.config(inner, outer)
                self.assertIsNone(namespace['_2zone_cont_calc_dmass'](cfg))
                self.assertAlmostEqual(cfg['zone_1_dust_mass'] + cfg['zone_2_dust_mass'], 0.009)
                # Independently reconstruct each boundary density by quadrature
                # of the physical (unnormalised) radial power law.
                integral_in = quad(lambda r: r ** (inner + 1), 10, 20)[0]
                integral_out = quad(lambda r: r ** (outer + 1), 20, 100)[0]
                sigma_in = cfg['zone_1_dust_mass'] * 20 ** inner / (2*np.pi*integral_in)
                sigma_out = cfg['zone_2_dust_mass'] * 20 ** outer / (2*np.pi*integral_out)
                np.testing.assert_allclose(sigma_in, sigma_out, rtol=1e-9, atol=0)
                self.assertEqual(cfg['zone_1_Rout'], cfg['zone_2_Rin'])
        cfg = self.config()
        namespace['_2zone_cont_calc_dmass'](cfg)
        self.assertAlmostEqual(cfg['zone_1_dust_mass'], 0.001)
        self.assertAlmostEqual(cfg['zone_2_dust_mass'], 0.008)

    def test_invalid_inputs(self):
        for key, value in [('disk_Rmid', 10), ('disk_Rmid', 100),
                           ('disk_total_dust_mass', -1), ('zone_1_Rin', 0),
                           ('zone_2_surface_density_exp', np.nan)]:
            cfg = self.config()
            cfg[key] = value
            with self.assertRaises(ValueError):
                namespace['_2zone_cont_calc_dmass'](cfg)

    def test_tapered_boundary_density(self):
        for slope, p2, rc in [(-1, -1, 40), (-2, -0.5, 15), (0.5, 0, 60)]:
            cfg = self.config(-2, slope)
            cfg.update(zone_2_type=2, zone_2_Rc=rc)
            cfg['zone_2_-gamma_exp'] = p2
            namespace['_2zone_cont_calc_dmass'](cfg)
            q = 2 + p2
            # Reconstruct density using the unnormalised physical profile.
            j1 = np.log(20 / 10)
            j2 = quad(lambda r: r**(slope+1)*np.exp(-(r/rc)**q),
                      20, 100, epsabs=0, epsrel=1e-10)[0]
            sigma1 = cfg['zone_1_dust_mass'] * 20**-2 / j1
            sigma2 = cfg['zone_2_dust_mass'] * 20**slope * np.exp(-(20/rc)**q) / j2
            np.testing.assert_allclose(sigma1, sigma2, rtol=1e-9)
            self.assertAlmostEqual(cfg['zone_1_dust_mass']+cfg['zone_2_dust_mass'], 0.009)

    def test_tapered_invalid_parameters(self):
        for rc, p2 in [(0, -1), (np.nan, -1), (40, -2), (40, np.inf)]:
            cfg = self.config()
            cfg.update(zone_2_type=2, zone_2_Rc=rc)
            cfg['zone_2_-gamma_exp'] = p2
            with self.assertRaises(ValueError):
                namespace['_2zone_cont_calc_dmass'](cfg)

    def test_flag(self):
        tree = ast.parse((ROOT / 'SMAC3.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and n.value.args
                    and isinstance(n.value.args[0], ast.Constant)
                    and n.value.args[0].value == '--2ZONE_CONT_LHC')
        parser = argparse.ArgumentParser()
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<flag>', 'exec'), {'p': parser})
        self.assertFalse(parser.parse_args([]).two_zone_cont_lhc)
        self.assertTrue(parser.parse_args(['--2ZONE_CONT_LHC']).two_zone_cont_lhc)

    def test_writer_trial_and_final_output(self):
        base = self.config()
        template = dict(number_of_zones='2', zone_1_type='1', zone_2_type='1',
                        zone_1_edge='0', zone_2_edge='0', zone_1_Rref='10',
                        zone_1_Rout='15', zone_2_Rin='15',
                        zone_1_dust_mass='0.004', zone_2_dust_mass='0.005')
        template.update({key: value for key, value in base.items() if key.startswith('zone_')})

        class FakeParaFile:
            def __init__(self, path):
                self.params = template.copy()
            def set_param(self, key, value):
                self.params[key] = value
            def save(self, path):
                path.write_text(json.dumps(self.params))

        namespace['ParaFile'] = FakeParaFile
        with tempfile.TemporaryDirectory() as directory:
            for relative in ('trials/config_1', 'smac_results'):
                outdir = Path(directory) / relative
                path = namespace['write_mcfost_paramfile'](base, {}, outdir, two_zone_cont_lhc=True)
                written = json.loads(path.read_text())
                self.assertEqual(written['zone_1_Rout'], 20)
                self.assertEqual(written['zone_2_Rin'], 20)
                self.assertAlmostEqual(written['zone_1_dust_mass'], 0.001)
                recorded = json.loads((outdir/'config_used.json').read_text())['cfg']
                self.assertEqual(recorded['zone_2_dust_mass'], written['zone_2_dust_mass'])
                self.assertNotIn('zone_1_dust_mass', base)
            disabled = {k: v for k, v in base.items() if k.startswith('zone_')}
            path = namespace['write_mcfost_paramfile'](disabled, {}, Path(directory)/'disabled')
            self.assertEqual(json.loads(path.read_text())['zone_1_dust_mass'], '0.004')
            for override in ({'zone_1_type': 2}, {'zone_2_edge': 1}, {'zone_1_dust_mass': 0.01}):
                with self.assertRaises(ValueError):
                    namespace['write_mcfost_paramfile'](dict(base, **override), {},
                                                       Path(directory)/'invalid', two_zone_cont_lhc=True)
            template.update(zone_2_type='2', zone_2_Rc='40')
            template['zone_2_-gamma_exp'] = '-1'
            for relative in ('trials/tapered', 'tapered_final'):
                path = namespace['write_mcfost_paramfile'](base, {}, Path(directory)/relative,
                                                         two_zone_cont_lhc=True)
                written = json.loads(path.read_text())
                expected = dict(base, zone_2_type=2, zone_2_Rc=40)
                expected['zone_2_-gamma_exp'] = -1
                namespace['_2zone_cont_calc_dmass'](expected)
                self.assertEqual(written['zone_2_type'], '2')
                self.assertAlmostEqual(written['zone_2_dust_mass'], expected['zone_2_dust_mass'])

    def test_tie_last_zone_before_mass_split(self):
        for count in (1, 2):
            template = dict(number_of_zones=str(count))
            for zone in range(1, count + 1):
                template.update({f'zone_{zone}_type': '1' if zone == 1 else '2',
                                 f'zone_{zone}_edge': '0',
                                 f'zone_{zone}_surface_density_exp': '-1',
                                 f'zone_{zone}_-gamma_exp': '-0.3',
                                 f'zone_{zone}_Rc': '40'})
            template.update(zone_1_Rin='10', zone_1_Rref='10', zone_2_Rout='100')
            template.update(zone_1_Rout='20', zone_2_Rin='20',
                            zone_1_dust_mass='0.001', zone_2_dust_mass='0.008')

            class FakeParaFile:
                def __init__(self, path):
                    self.params = template.copy()
                def set_param(self, key, value):
                    self.params[key] = value
                def save(self, path):
                    path.write_text(json.dumps(self.params))

            namespace['ParaFile'] = FakeParaFile
            for sampled_slope in (False, True):
                cfg = dict(disk_Rmid=20, disk_total_dust_mass=0.009) if count == 2 else {}
                slope_key = f'zone_{count}_surface_density_exp'
                taper_key = f'zone_{count}_-gamma_exp'
                cfg[taper_key] = -0.7  # Explicitly overridden when the flag is on.
                if sampled_slope:
                    cfg[slope_key] = -1.5
                original = cfg.copy()
                with tempfile.TemporaryDirectory() as directory:
                    for relative in ('trials/test', 'smac_results'):
                        outdir = Path(directory)/relative
                        path = namespace['write_mcfost_paramfile'](
                            cfg, {}, outdir, two_zone_cont_lhc=count == 2,
                            tapered_edge_p1_eq_p2=True)
                        written = json.loads(path.read_text())
                        expected_slope = -1.5 if sampled_slope else -1
                        self.assertEqual(float(written[taper_key]), expected_slope)
                        recorded = json.loads((outdir/'config_used.json').read_text())['cfg']
                        self.assertEqual(float(recorded[taper_key]), expected_slope)
                        self.assertEqual(cfg, original)
                        if count == 2:
                            self.assertEqual(written['zone_1_-gamma_exp'], '-0.3')
                            expected = self.config(-1, expected_slope)
                            expected.update(zone_2_type=2, zone_2_Rc=40)
                            expected['zone_2_-gamma_exp'] = expected_slope
                            namespace['_2zone_cont_calc_dmass'](expected)
                            self.assertAlmostEqual(written['zone_2_dust_mass'], expected['zone_2_dust_mass'])
                    path = namespace['write_mcfost_paramfile'](
                        cfg, {}, Path(directory)/'disabled', two_zone_cont_lhc=count == 2)
                    self.assertEqual(json.loads(path.read_text())[taper_key], -0.7)


if __name__ == '__main__':
    unittest.main()
