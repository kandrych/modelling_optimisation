"""Exercise launch commands with Python/module execution replaced by harmless stubs."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run_model


ROOT = Path(__file__).resolve().parents[1]


def optimisation_arguments(arguments):
    # Execute only the real argument parser, stopping before optimisation setup.
    tree = ast.parse((ROOT / 'SMAC3.py').read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    parser_statements = []
    for node in main.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'p':
                parser_statements.append(node)
                continue
            if node.targets[0].id == 'args':
                break
        if parser_statements:
            parser_statements.append(node)
    namespace = dict(argparse=argparse, positive_seconds=float, timeout_cost=str)
    exec(compile(ast.Module(body=parser_statements, type_ignores=[]), 'SMAC3.py parser', 'exec'), namespace)
    return namespace['p'].parse_args(arguments)


class LaunchScriptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='launch checks ')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.record = self.work / 'command.json'
        self.stub = self.work / 'capture.py'
        self.stub.write_text(
            'import json, os, pathlib, sys\n'
            'pathlib.Path(os.environ["LAUNCH_RECORD"]).write_text(json.dumps({\n'
            '    "arguments": sys.argv[1:],\n'
            '    "omp": os.environ["OMP_NUM_THREADS"],\n'
            '    "blas": os.environ["OPENBLAS_NUM_THREADS"],\n'
            '}))\n')
        self.shell_setup = self.work / 'bash_environment.sh'
        self.shell_setup.write_text('''module() { :; }
source() {
    # Keep the production setup logic, but do not activate a site-specific venv.
    if [[ "$1" == */bin/activate ]]; then return 0; fi
    builtin source "$@"
}
python() { command "$LAUNCH_PYTHON" -B "$LAUNCH_CAPTURE" "$@"; }
export -f module source python
''')

    def launch(self, script, cpus, *arguments, **overrides):
        environment = dict(os.environ)
        for key in ('N_WORKERS', 'WORK_ROOT', 'CONFIG_SPACE', 'FIDELITY_CONFIG',
                    'WARMSTART', 'DATA_ROOT', 'OUTPUT_DIR', 'PARA_FILE',
                    'UNRESOLVED_CORRECTION_RADIUS_PX'):
            environment.pop(key, None)
        environment.update(PROJECT_ROOT=str(ROOT), SLURM_SUBMIT_DIR=str(self.work),
                           SLURM_CPUS_PER_TASK=str(cpus), BASH_ENV=str(self.shell_setup),
                           LAUNCH_PYTHON=sys.executable, LAUNCH_CAPTURE=str(self.stub),
                           LAUNCH_RECORD=str(self.record))
        environment.update(overrides)
        return subprocess.run(['bash', str(ROOT / script), *arguments], env=environment,
                              text=True, capture_output=True, timeout=20)

    def test_optimisation_and_warmstart_arguments_and_thread_limits(self):
        for script, cpus in (('run_scripts/run_smac.q', 21), ('run_scripts/run_smac_warmstart.q', 58)):
            with self.subTest(script=script):
                result = self.launch(script, cpus, '--n-trials', '7',
                                     CONFIG_SPACE=str(self.work / 'chosen config.yaml'),
                                     WARMSTART=str(self.work / 'previous run'))
                self.assertEqual(result.returncode, 0, result.stderr)
                command = json.loads(self.record.read_text())
                self.assertEqual(command['arguments'][0], str(ROOT / 'SMAC3.py'))
                args = optimisation_arguments(command['arguments'][1:])
                self.assertEqual(args.n_trials, 7)
                self.assertEqual(args.n_workers, 10)
                self.assertEqual(args.config_space, str(self.work / 'chosen config.yaml'))
                self.assertEqual(args.fidelity_config, str(self.work / 'fidelity.txt'))
                self.assertEqual(int(command['omp']), cpus // args.n_workers)
                self.assertEqual(command['blas'], '1')
                if 'warmstart' in script:
                    self.assertEqual(args.warmstart, str(self.work / 'previous run'))
                    self.assertTrue(args.puffed_up_rim)

    def test_single_model_and_old_launch_filenames_use_supported_driver(self):
        for script, data_root, corrected in (
                ('run_scripts/run_model.q', 'demo_ozstar', False),
                ('run_scripts/run_smac_chromatic_1_simulation.q', 'demo_ozstar', True),
                ('scripts/run_smac.q', 'ar_pup_ozstar', False)):
            with self.subTest(script=script):
                result = self.launch(script, 64, '--stage', 'chosen_stage',
                                     UNRESOLVED_CORRECTION_RADIUS_PX='3')
                self.assertEqual(result.returncode, 0, result.stderr)
                command = json.loads(self.record.read_text())
                self.assertEqual(command['arguments'][0], str(ROOT / 'run_model.py'))
                args = run_model.parse_args(command['arguments'][1:])
                self.assertEqual(args.stage, 'chosen_stage')
                self.assertEqual(args.data_root, data_root)
                self.assertEqual(args.correct_unresolved_polarimetry, corrected)
                self.assertEqual(command['omp'], '64')

    def test_invalid_worker_count_and_missing_warmstart_stop_before_python(self):
        for script, overrides in (
                ('run_scripts/run_smac.q', {'N_WORKERS': '22'}),
                ('run_scripts/run_smac_warmstart.q', {}),
                ('run_scripts/run_smac_chromatic_1_simulation.q', {})):
            with self.subTest(script=script):
                result = self.launch(script, 21, **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.record.exists())

    def test_worker_override_cannot_bypass_the_thread_allocation(self):
        for arguments in (('--n-workers', '21'), ('--n-workers=21',)):
            result = self.launch('run_scripts/run_smac.q', 21, *arguments)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Set N_WORKERS', result.stderr)
            self.assertFalse(self.record.exists())


if __name__ == '__main__':
    unittest.main()
