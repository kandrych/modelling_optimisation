"""Final regenerations must not reuse previous MCFOST outputs; no simulations."""
import __future__
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def extract(relative_path, name, namespace):
    tree = ast.parse((ROOT / relative_path).read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), relative_path, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


prepare_results = extract("SMAC3.py", "prepare_final_results_dir",
                          {"Path": Path, "tempfile": tempfile})


class FinalRegenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            return prepare_results(self.root)

    def archives(self):
        return sorted((self.root / "smac_results_history").glob("run_*/smac_results"))

    def simulate(self, output, products, generation, fail_at=None):
        images = []

        def fake_mcfost(param_path, workdir, options=None, logfile=None):
            self.assertEqual(Path(param_path), output / "model.para")
            self.assertEqual(workdir, output)
            if options and options[0] == "-img":
                wavelength = float(options[1])
                images.append(wavelength)
                if wavelength == fail_at:
                    raise RuntimeError("interrupted image generation")
                folder = workdir / f"data_{wavelength}"
                folder.mkdir()
                (folder / "image.fits").write_text(generation)
            else:
                folder = workdir / ("data_disk" if options else "data_th")
                folder.mkdir()
                (folder / "output.fits").write_text(generation)

        run_mcfost = extract("lib/obriy_mcfost.py", "run_mcfost",
                             {"os": os, "run_mcfost_safe": fake_mcfost})
        (output / "model.para").write_text(generation)
        with redirect_stdout(io.StringIO()):
            run_mcfost({"products": products}, output / "model.para", output)
        return images

    def test_first_run_is_empty_and_keeps_standard_layout(self):
        output = self.prepare()
        self.assertEqual(output, self.root / "smac_results")
        self.assertEqual(list(output.iterdir()), [])
        self.assertFalse((self.root / "smac_results_history").exists())

    def test_previous_results_and_postprocessing_are_preserved_intact(self):
        output = self.prepare()
        files = {"incumbent.json": "old config", "trials.csv": "old history",
                 "figures/comparison.png": "old figure",
                 "after_optimisation/notes.txt": "user notes",
                 "data_0.82/image.fits": "old image"}
        for name, contents in files.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        trial = self.root / "trials" / "config_1"
        trial.mkdir(parents=True)
        (trial / "keep.txt").write_text("trial")
        self.prepare()
        self.assertEqual(list(output.iterdir()), [])
        archive, = self.archives()
        for name, contents in files.items():
            self.assertEqual((archive / name).read_text(), contents)
        self.assertEqual((trial / "keep.txt").read_text(), "trial")
        (output / "incumbent.json").write_text("second config")
        self.prepare()
        self.assertEqual(len(self.archives()), 2)
        self.assertEqual((archive / "incumbent.json").read_text(), "old config")

    def test_new_model_regenerates_images_for_both_interferometry_modes(self):
        for mode in ("vis2_1perband", "vis2_chromatic"):
            with self.subTest(mode=mode):
                products = [mode, "pdi_V", "pdi_I", "pdi_H"]
                output = self.prepare()
                old_images = self.simulate(output, products, "old model")
                output = self.prepare()
                new_images = self.simulate(output, products, "new model")
                self.assertEqual(new_images, old_images)
                self.assertTrue({0.55, 0.82, 1.63, 1.65, 2.2, 3.5, 10.0} <= set(new_images))
                # Overlapping PDI/chromatic wavelengths are still computed once
                # within a single run, but never taken from an earlier model.
                self.assertEqual(len(new_images), len(set(new_images)))
                for path in output.glob("data_*/*.fits"):
                    self.assertEqual(path.read_text(), "new model")

    def test_retry_does_not_reuse_outputs_from_an_interrupted_run(self):
        output = self.prepare()
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.simulate(output, ["vis2_1perband"], "partial model", fail_at=2.2)
        self.assertTrue((output / "data_1.65/image.fits").exists())
        output = self.prepare()
        images = self.simulate(output, ["vis2_1perband"], "retry model")
        self.assertEqual(images, [1.65, 2.2, 3.5, 10.0])
        archive, = self.archives()
        self.assertEqual((archive / "data_1.65/image.fits").read_text(), "partial model")
        self.assertEqual((output / "data_1.65/image.fits").read_text(), "retry model")

    def test_unexpected_file_or_symlink_is_rejected_without_modifying_it(self):
        output = self.root / "smac_results"
        output.write_text("keep")
        with self.assertRaisesRegex(ValueError, "regular results directory"):
            self.prepare()
        self.assertEqual(output.read_text(), "keep")
        output.unlink()
        target = self.root / "other_results"
        target.mkdir()
        (target / "keep.txt").write_text("keep")
        output.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "regular results directory"):
            self.prepare()
        self.assertTrue(output.is_symlink())
        self.assertEqual((target / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
