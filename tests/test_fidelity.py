import ast
import tempfile
import unittest
from pathlib import Path

from lib.obriy_fidelity import load_fidelity_config, map_budget_to_fidelity, fidelity_products


class FidelityTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fidelity.txt'
            path.write_text(text)
            return load_fidelity_config(path)

    def test_example_and_budget_roundoff(self):
        settings = load_fidelity_config(Path(__file__).resolve().parents[1]/'fidelity.txt')
        self.assertEqual([s['budget'] for s in settings['stages']], [1, 3, 9])
        self.assertEqual(map_budget_to_fidelity(3.00000000001, settings)['stage'], 'sed_pdi')
        self.assertIn('alma', fidelity_products(settings))
        with self.assertRaises(ValueError):
            map_budget_to_fidelity(14.1, settings)

    def test_single_stage(self):
        settings = self.load('fidelity_stages:\n- name: only\n  products: [alma]\n')
        self.assertEqual(settings['stages'][0]['budget'], 1)
        self.assertEqual(fidelity_products(settings), ['alma'])

    def test_custom_eta_and_union(self):
        settings = self.load('eta: 2\nfidelity_stages:\n- name: first\n  products: [alma]\n- name: second\n  products: [sed]\n')
        self.assertEqual([s['budget'] for s in settings['stages']], [1, 2])
        self.assertEqual(fidelity_products(settings), ['alma', 'sed'])

    def test_invalid_files(self):
        cases = ['[]', 'fidelity_stages: []',
                 'eta: 1\nfidelity_stages: [{name: s, products: [sed]}]',
                 'fidelity_stages: [{name: s, products: [unknown]}]',
                 'fidelity_stages: [{name: s, products: [sed, sed]}]',
                 'fidelity_stages: [{name: s, products: [vis2_1perband, vis2_chromatic]}]',
                 'fidelity_stages: [{name: s, products: [sed]}, {name: s, products: [alma]}]']
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_legacy_map_is_commented(self):
        source = (Path(__file__).resolve().parents[1]/'SMAC3.py').read_text()
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name == 'map_budget_to_fidelity' for n in tree.body))
        self.assertIn('# def map_budget_to_fidelity(', source)


if __name__ == '__main__':
    unittest.main()
