"""Data loading and leak-free patient-level splitting utilities for BraTS 2020."""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 256)
CLASS_NAMES: Tuple[str, str, str] = ("Necrotic", "Enhancing Tumor", "Edema")

# Matches Kaggle awsaf49/brats2020-training-data HDF5 slice filenames:
# e.g. volume_100_slice_120.h5 -> patient/volume group "volume_100".
_VOLUME_PATTERN = re.compile(r"^(volume_\d+)_slice_\d+\.h5$", re.IGNORECASE)
_BRATS_PATTERN = re.compile(r"^(BraTS\d+_Training_\d+)", re.IGNORECASE)


def extract_patient_id(filepath: str) -> str:
    """Extract the patient/volume identifier from a BraTS 2D slice filename."""
    name = os.path.basename(filepath)
    match = _VOLUME_PATTERN.match(name)
    if match:
        return match.group(1).lower()
    match = _BRATS_PATTERN.match(name)
    if match:
        return match.group(1)
    raise ValueError(
        f"Cannot parse patient ID from filename '{name}'. "
        "Expected pattern 'volume_<id>_slice_<idx>.h5'."
    )


def discover_h5_slices(data_dir: str) -> List[str]:
    """Return sorted list of HDF5 slice paths inside data_dir."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 slice files found in '{data_dir}'.")
    return files


def split_by_patient(
    file_paths: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Split slice file paths into train and validation sets grouped by patient ID.

    Prevents adjacent MRI slices from the same volume appearing in both splits.
    """
    if not file_paths:
        raise ValueError("file_paths must be non-empty.")

    paths = list(file_paths)
    groups = [extract_patient_id(p) for p in paths]
    unique_patients = set(groups)
    if len(unique_patients) < 2:
        raise ValueError(
            f"Need at least 2 distinct patients for GroupShuffleSplit, found {len(unique_patients)}."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(paths, groups=groups))

    train_files = sorted(paths[i] for i in train_idx)
    val_files = sorted(paths[i] for i in val_idx)

    train_patients = {groups[i] for i in train_idx}
    val_patients = {groups[i] for i in val_idx}
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(f"Data leakage detected across patient split: {overlap}")

    summary = {
        "total_slices": len(paths),
        "total_patients": len(unique_patients),
        "train_slices": len(train_files),
        "val_slices": len(val_files),
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
    }
    return train_files, val_files, summary


def load_raw_h5_slice(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw 4-channel MRI slice and 3-channel tumor mask from an HDF5 file."""
    with h5py.File(filepath, "r") as handle:
        if "image" not in handle or "mask" not in handle:
            raise KeyError(
                f"Expected 'image' and 'mask' datasets in '{filepath}', "
                f"found {list(handle.keys())}."
            )
        image = np.asarray(handle["image"][:], dtype=np.float32)
        mask = np.asarray(handle["mask"][:], dtype=np.float32)
    return image, mask


def build_data_generator(
    file_paths: Sequence[str],
    batch_size: int = 16,
    img_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    shuffle: bool = False,
    seed: int = 42,
):
    """Create a Keras Sequence data generator resizing MRI slices and masks on the fly."""
    import tensorflow as tf

    class BraTSDataGenerator(tf.keras.utils.Sequence):
        """Keras Sequence yielding resized (256, 256, 4) MRI and (256, 256, 3) mask batches."""

        def __init__(
            self,
            paths: Sequence[str],
            batch_sz: int,
            target_size: Tuple[int, int],
            do_shuffle: bool,
            rng_seed: int,
        ) -> None:
            self.file_paths = list(paths)
            self.batch_size = batch_sz
            self.img_size = target_size
            self.shuffle = do_shuffle
            self.rng = np.random.default_rng(rng_seed)
            self.indices = np.arange(len(self.file_paths))
            if self.shuffle:
                self.rng.shuffle(self.indices)

        def __len__(self) -> int:
            return int(np.ceil(len(self.file_paths) / self.batch_size))

        def on_epoch_end(self) -> None:
            if self.shuffle:
                self.rng.shuffle(self.indices)

        def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
            batch_ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
            batch_files = [self.file_paths[i] for i in batch_ids]

            x_batch = np.empty((len(batch_files), *self.img_size, 4), dtype=np.float32)
            y_batch = np.empty((len(batch_files), *self.img_size, 3), dtype=np.float32)

            for idx, path in enumerate(batch_files):
                img_raw, mask_raw = load_raw_h5_slice(path)
                x_batch[idx] = tf.image.resize(img_raw, self.img_size, method="bilinear").numpy()
                y_batch[idx] = tf.image.resize(mask_raw, self.img_size, method="nearest").numpy()

            return x_batch, y_batch

    return BraTSDataGenerator(file_paths, batch_size, img_size, shuffle, seed)
