"""Packaging, local data and fail-closed driver checks; no real data downloads."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from importlib.resources import files
import numpy as np
from simba import data_tools, paths, run_matrix
from simba.demo import run


class StandaloneTests(unittest.TestCase):
    def test_included_demo_excludes_seeds_and_calibrates_on_validation(self):
        data = json.loads(files("simba").joinpath("example_data.json").read_text())
        result = run(data)
        self.assertEqual(len(result["queries"]), len(data["test"]))
        self.assertTrue(np.isfinite(result["validation_calibration"]["prior_scale"]))
        self.assertTrue(0 <= result["mean_recall"] <= 1)
        for query in result["queries"]:
            ids = query["recommendations"]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertFalse(set(ids) & set(query["seeds"]))

    def test_paths_use_explicit_root_and_per_dataset_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "food").mkdir()
            (root / "override").mkdir()
            with patch.dict(os.environ, {"SIMBA_DATA_DIR": temp, "BR_DIR": str(root / "override")}):
                self.assertEqual(paths.dataset_dir("food"), str((root / "food").resolve()))
                self.assertEqual(paths.dataset_dir("food", "BR_DIR"), str((root / "override").resolve()))
                with self.assertRaises(FileNotFoundError):
                    paths.dataset_dir("spotify")

    def test_missing_data_and_hash_inventory(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            ready, _ = data_tools.check(["food"], root)
            self.assertFalse(ready)
            (root / "food").mkdir()
            for filename in data_tools.required_files("food"):
                (root / "food" / filename).write_text("example\n")
            ready, inventory = data_tools.check(["food"], root, hashes=True)
            self.assertTrue(ready)
            self.assertEqual(len(inventory), 6)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in inventory))

    def test_failed_download_does_not_leave_a_complete_file(self):
        class Interrupted(io.BytesIO):
            def read(self, size=-1):
                raise OSError("interrupted transfer")
        with tempfile.TemporaryDirectory() as temp, patch.object(data_tools, "urlopen", return_value=Interrupted()), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                data_tools.fetch("food", Path(temp))
            self.assertFalse((Path(temp) / "food/bundle_item.csv").exists())
            self.assertFalse(list(Path(temp).rglob("*.part")))

    def test_parser_requires_all_measured_methods(self):
        text = "\n".join(f"{name} .1 .2 .3 .4" for name in run_matrix.METHODS)
        self.assertEqual(len(run_matrix.parse_rows(text)), 7)
        with self.assertRaises(ValueError):
            run_matrix.parse_rows("Ours .1 .2 .3 .4")
        with self.assertRaises(ValueError):
            run_matrix.parse_rows(text.replace(".3", "nan"))

    def test_failed_worker_cannot_publish_partial_rows(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run_matrix.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "Ours .1 .2 .3 .4", "failure")):
            with self.assertRaises(RuntimeError):
                run_matrix.run_worker("food", {}, Path(temp) / "food.log")

    def test_driver_only_writes_current_measured_methods(self):
        rows = {name: [.1, .2, .3, .4] for name in run_matrix.METHODS}
        with tempfile.TemporaryDirectory() as temp, patch.object(run_matrix, "run_worker", return_value=rows), contextlib.redirect_stdout(io.StringIO()):
            run_matrix.main(["food", "--seeds", "2", "--output", temp])
            text = (Path(temp) / "measured_matrix.csv").read_text()
            self.assertNotIn("CLHE", text)
            self.assertNotIn("RaMen", text)
            self.assertEqual(len(text.splitlines()), 8)

    def test_driver_defaults_to_all_datasets(self):
        rows = {name: [.1, .2, .3, .4] for name in run_matrix.METHODS}
        with tempfile.TemporaryDirectory() as temp, patch.object(run_matrix, "run_worker", return_value=rows) as worker, contextlib.redirect_stdout(io.StringIO()):
            run_matrix.main(["--seeds", "1", "--output", temp])
            self.assertEqual(worker.call_count, 8)


if __name__ == "__main__":
    unittest.main()
