"""Unit tests for patient-level data splitting, HDF5 loading, and segmentation metrics."""

from __future__ import annotations

import os
import tempfile
import unittest

import h5py
import numpy as np

from src.data import (
    discover_h5_slices,
    extract_patient_id,
    load_raw_h5_slice,
    split_by_patient,
)
from src.metrics import compute_per_class_metrics, format_evaluation_report


class TestBraTSPipeline(unittest.TestCase):
    def test_extract_patient_id(self) -> None:
        self.assertEqual(extract_patient_id("/data/volume_100_slice_120.h5"), "volume_100")
        self.assertEqual(extract_patient_id(r"C:\brats\Volume_1_Slice_0.h5"), "volume_1")
        with self.assertRaises(ValueError):
            extract_patient_id("random_image.h5")

    def test_patient_level_split_zero_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for patient_id in range(1, 21):
                for slice_idx in range(5):
                    path = os.path.join(tmpdir, f"volume_{patient_id}_slice_{slice_idx}.h5")
                    with h5py.File(path, "w") as f:
                        f.create_dataset("image", data=np.zeros((240, 240, 4), dtype=np.float32))
                        f.create_dataset("mask", data=np.zeros((240, 240, 3), dtype=np.float32))

            all_files = discover_h5_slices(tmpdir)
            self.assertEqual(len(all_files), 100)

            train_files, val_files, summary = split_by_patient(
                all_files, test_size=0.2, random_state=42
            )
            train_patients = {extract_patient_id(p) for p in train_files}
            val_patients = {extract_patient_id(p) for p in val_files}

            self.assertEqual(len(train_patients & val_patients), 0)
            self.assertEqual(summary["total_patients"], 20)
            self.assertEqual(summary["train_slices"] + summary["val_slices"], 100)

            img, msk = load_raw_h5_slice(train_files[0])
            self.assertEqual(img.shape, (240, 240, 4))
            self.assertEqual(msk.shape, (240, 240, 3))

    def test_metrics_exactness_and_report(self) -> None:
        y_true = np.zeros((2, 8, 8, 3), dtype=np.float32)
        y_pred = np.zeros((2, 8, 8, 3), dtype=np.float32)

        # Class 0 (Necrotic): exact overlap on 4 pixels -> Dice=1.0, IoU=1.0
        y_true[0, 0:2, 0:2, 0] = 1.0
        y_pred[0, 0:2, 0:2, 0] = 1.0

        # Class 1 (Enhancing Tumor): TP=2, FN=2, FP=2 -> Sens=0.5, Prec=0.5, Dice=0.5, IoU=1/3
        y_true[0, 0:2, 0:2, 1] = 1.0
        y_pred[0, 0:2, 1:3, 1] = 1.0

        # Class 2 (Edema): all empty -> handled cleanly without division by zero
        metrics = compute_per_class_metrics(y_true, y_pred, threshold=0.5)

        self.assertAlmostEqual(metrics["Necrotic"]["global_dice"], 1.0, places=5)
        self.assertAlmostEqual(metrics["Enhancing Tumor"]["global_dice"], 0.5, places=5)
        self.assertAlmostEqual(metrics["Enhancing Tumor"]["global_iou"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(metrics["Enhancing Tumor"]["sensitivity"], 0.5, places=5)
        self.assertAlmostEqual(metrics["Edema"]["global_dice"], 1.0, places=5)

        report = format_evaluation_report(
            metrics=metrics,
            total_params=10_116_079,
            split_summary={
                "train_slices": 80,
                "val_slices": 20,
                "train_patients": 16,
                "val_patients": 4,
            },
            epochs=50,
            batch_size=16,
            learning_rate=1e-4,
        )
        self.assertIn("10,116,079", report)
        self.assertNotIn("{len(", report)
        self.assertIn("patient-level GroupShuffleSplit", report)


if __name__ == "__main__":
    unittest.main()
