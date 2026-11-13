"""Pure NumPy/scikit-learn segmentation metrics for BraTS 2020."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

from src.data import CLASS_NAMES

EPS = 1e-7


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, Dict[str, float]]:
    """Compute global pixel confusion counts and slice/dataset Dice, IoU, sensitivity, precision.

    Args:
        y_true: Ground-truth tensor with shape (N, H, W, C).
        y_pred: Predicted probabilities with shape (N, H, W, C).
        threshold: Binarization threshold applied to probabilities and targets.
        class_names: Ordered tumor subregion names matching C.
    """
    if y_true.ndim != 4 or y_pred.ndim != 4:
        raise ValueError(
            f"Expected 4D tensors (N, H, W, C); got y_true={y_true.shape}, y_pred={y_pred.shape}."
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape} vs y_pred={y_pred.shape}.")
    if y_true.shape[-1] != len(class_names):
        raise ValueError(
            f"Channel count {y_true.shape[-1]} does not match class_names ({len(class_names)})."
        )

    results: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(class_names):
        gt_bin = (y_true[..., idx] > threshold).astype(np.uint8)
        pr_bin = (y_pred[..., idx] > threshold).astype(np.uint8)

        cm = confusion_matrix(gt_bin.ravel(), pr_bin.ravel(), labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        global_dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
        global_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0

        # Per-slice metrics averaged over slices containing either foreground or prediction
        slice_dice: List[float] = []
        slice_iou: List[float] = []
        for s_idx in range(gt_bin.shape[0]):
            gt_s = gt_bin[s_idx]
            pr_s = pr_bin[s_idx]
            inter = float(np.sum(gt_s & pr_s))
            gt_sum = float(np.sum(gt_s))
            pr_sum = float(np.sum(pr_s))
            if gt_sum + pr_sum == 0.0:
                continue
            union = gt_sum + pr_sum - inter
            slice_iou.append(inter / (union + EPS))
            slice_dice.append((2.0 * inter) / (gt_sum + pr_sum + EPS))

        results[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "sensitivity": float(sensitivity),
            "precision": float(precision),
            "global_dice": float(global_dice),
            "global_iou": float(global_iou),
            "mean_slice_dice": float(np.mean(slice_dice)) if slice_dice else 1.0,
            "std_slice_dice": float(np.std(slice_dice)) if slice_dice else 0.0,
            "mean_slice_iou": float(np.mean(slice_iou)) if slice_iou else 1.0,
            "std_slice_iou": float(np.std(slice_iou)) if slice_iou else 0.0,
            "nonempty_slices": len(slice_dice),
        }

    return results


def format_evaluation_report(
    metrics: Dict[str, Dict[str, float]],
    total_params: int,
    split_summary: Dict[str, int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> str:
    """Format a plain-text, reproducible summary table with exact model and split counts."""
    lines = [
        "BraTS 2020 Multi-Class Segmentation Evaluation Report",
        "=" * 58,
        f"Architecture:       U-Net + EfficientNetB0 (encoder_weights=None)",
        f"Total Parameters:   {total_params:,}",
        f"Training Config:    epochs={epochs}, batch_size={batch_size}, lr={learning_rate}",
        (
            "Dataset Split:      patient-level GroupShuffleSplit | "
            f"train={split_summary.get('train_slices', 0)} slices "
            f"({split_summary.get('train_patients', 0)} patients) | "
            f"val={split_summary.get('val_slices', 0)} slices "
            f"({split_summary.get('val_patients', 0)} patients)"
        ),
        "-" * 58,
    ]

    dice_vals: List[float] = []
    iou_vals: List[float] = []
    sens_vals: List[float] = []
    prec_vals: List[float] = []

    for name, m in metrics.items():
        dice_vals.append(m["global_dice"])
        iou_vals.append(m["global_iou"])
        sens_vals.append(m["sensitivity"])
        prec_vals.append(m["precision"])
        lines.append(
            f"{name:<16} | Global Dice={m['global_dice']:.4f} | "
            f"Global IoU={m['global_iou']:.4f} | "
            f"Sens={m['sensitivity']:.4f} | Prec={m['precision']:.4f}"
        )

    lines.extend(
        [
            "-" * 58,
            (
                f"Macro Average    | Global Dice={np.mean(dice_vals):.4f} | "
                f"Global IoU={np.mean(iou_vals):.4f} | "
                f"Sens={np.mean(sens_vals):.4f} | Prec={np.mean(prec_vals):.4f}"
            ),
            (
                "Note: Background specificity is omitted from headline metrics because "
                "empty/background pixels account for ~99% of slice area."
            ),
        ]
    )
    return "\n".join(lines)
