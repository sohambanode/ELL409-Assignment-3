"""Data auditing and deterministic split tests using synthetic images."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from neural_tree.core import group_split, prepare_cats, stratified_subset, validate_manifest, corrupt_labels


class DataTests(unittest.TestCase):
    def test_noise_changes_exact_count_without_mutation(self):
        rows = [{"id": str(i), "label": i % 10} for i in range(100)]
        original = copy.deepcopy(rows)
        noisy = corrupt_labels(rows, 10, .2, 42)
        self.assertEqual(rows, original)
        self.assertEqual(noisy, corrupt_labels(rows, 10, .2, 42))
        self.assertEqual(sum(a["label"] != b["label"] for a, b in zip(rows, noisy)), 20)
        self.assertTrue(all(0 <= r["label"] < 10 for r in noisy))
    def test_groups_never_cross_splits_and_are_reproducible(self):
        rows = [{"id": str(i), "label": i % 2, "group": str(i // 2)} for i in range(100)]
        result = group_split(copy.deepcopy(rows), 42)
        self.assertEqual(result, group_split(copy.deepcopy(rows), 42))
        validate_manifest({"records": result, "classes": ["cat", "dog"], "root": "."})
        self.assertEqual({r["split"] for r in result}, {"train", "val", "test"})

    def test_balanced_subset(self):
        rows = [{"id": str(i), "label": i % 2} for i in range(100)]
        selected = stratified_subset(rows, 11, 7)
        self.assertEqual(len(selected), 11)
        self.assertEqual(sum(r["label"] for r in selected), 5)
        self.assertEqual(selected, stratified_subset(rows, 11, 7))
        with self.assertRaises(ValueError):
            stratified_subset(rows, 101)

    def test_reject_cross_split_duplicate(self):
        manifest = {"root": ".", "classes": ["x"], "records": [
            {"id": "1", "label": 0, "group": "same", "split": "train"},
            {"id": "2", "label": 0, "group": "same", "split": "test"}]}
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_decode_duplicates_conflicting_labels_and_hash_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(42)
            for folder in ("Cat", "Dog"):
                (root / folder).mkdir()
                for i in range(20):
                    Image.fromarray(rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)).save(root / folder / f"{i}.png")
            (root / "Cat" / "duplicate.png").write_bytes((root / "Cat" / "0.png").read_bytes())
            (root / "Dog" / "conflict.png").write_bytes((root / "Cat" / "1.png").read_bytes())
            (root / "Dog" / "broken.jpg").write_bytes(b"not an image")
            output = root / "manifest.json"
            prepare_cats(root, output, 42, "synthetic test fixture")
            manifest = json.loads(output.read_text())
            self.assertEqual(len(manifest["audit"]["excluded"]), 3)
            self.assertEqual(len(manifest["records"]), 40)
            validate_manifest(manifest, verify_files=True)
            images = {r["id"]: r for r in manifest["records"]}
            self.assertEqual(images["Cat/0.png"]["split"], images["Cat/duplicate.png"]["split"])
            (root / "Cat" / "0.png").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_manifest(manifest, verify_files=True)


if __name__ == "__main__":
    unittest.main()
