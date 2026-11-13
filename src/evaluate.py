"""Full-validation evaluation, confusion matrix plotting, and qualitative heatmaps."""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data import CLASS_NAMES, build_data_generator, discover_h5_slices, split_by_patient
from src.metrics import compute_per_class_metrics, format_evaluation_report
from src.model import build_unet_efficientnetb0


def evaluate_generator(
    model,
    val_gen,
    max_batches: Optional[int] = None,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Run inference over the full validation generator (or max_batches if specified)."""
    num_batches = len(val_gen) if max_batches is None else min(len(val_gen), max_batches)
    y_true_batches: List[np.ndarray] = []
    y_pred_batches: List[np.ndarray] = []

    for batch_idx in range(num_batches):
        x_batch, y_true_batch = val_gen[batch_idx]
        y_pred_batch = model.predict(x_batch, verbose=0)
        y_true_batches.append(y_true_batch)
        y_pred_batches.append(y_pred_batch)

    y_true_all = np.concatenate(y_true_batches, axis=0)
    y_pred_all = np.concatenate(y_pred_batches, axis=0)
    return compute_per_class_metrics(y_true_all, y_pred_all, threshold=threshold)


def plot_confusion_matrices(
    metrics: Dict[str, Dict[str, float]],
    output_path: str = "confusion_matrices.png",
) -> None:
    """Save 1x3 per-class pixel confusion matrices."""
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(15, 4))
    for idx, name in enumerate(CLASS_NAMES):
        m = metrics[name]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=np.int64)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        axes[idx].set_title(f"Confusion Matrix - {name}")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_predictions_heatmap(
    model,
    val_gen,
    num_samples: int = 6,
    output_path: str = "predictions_heatmap.png",
) -> None:
    """Save side-by-side ground truth vs predicted probability maps for validation slices."""
    fig, axes = plt.subplots(num_samples, 7, figsize=(20, 16))
    for sample_idx in range(min(num_samples, len(val_gen))):
        x_batch, y_true = val_gen[sample_idx]
        y_pred = model.predict(x_batch[0:1], verbose=0)

        axes[sample_idx, 0].imshow(x_batch[0, :, :, 0], cmap="gray")
        axes[sample_idx, 0].set_title(f"Sample {sample_idx + 1}\nT1 Image")
        axes[sample_idx, 0].axis("off")

        for c_idx, class_name in enumerate(CLASS_NAMES):
            axes[sample_idx, 1 + c_idx * 2].imshow(y_true[0, :, :, c_idx], cmap="hot", vmin=0, vmax=1)
            axes[sample_idx, 1 + c_idx * 2].set_title(f"{class_name}\nGround Truth")
            axes[sample_idx, 1 + c_idx * 2].axis("off")

            axes[sample_idx, 2 + c_idx * 2].imshow(y_pred[0, :, :, c_idx], cmap="hot", vmin=0, vmax=1)
            axes[sample_idx, 2 + c_idx * 2].set_title(f"{class_name}\nPrediction")
            axes[sample_idx, 2 + c_idx * 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained BraTS 2020 U-Net model.")
    parser.add_argument("--data-dir", default="BraTS2020_training_data/content/data")
    parser.add_argument("--weights", default="best_model_weights.h5")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--report-path", default="evaluation_report.txt")
    args = parser.parse_args()

    all_files = discover_h5_slices(args.data_dir)
    _, val_files, split_summary = split_by_patient(
        all_files, test_size=args.test_size, random_state=args.seed
    )
    val_gen = build_data_generator(val_files, batch_size=args.batch_size, shuffle=False)

    model = build_unet_efficientnetb0()
    model.load_weights(args.weights)

    metrics = evaluate_generator(model, val_gen, max_batches=args.max_batches)
    plot_confusion_matrices(metrics)
    plot_predictions_heatmap(model, val_gen)

    report = format_evaluation_report(
        metrics=metrics,
        total_params=model.count_params(),
        split_summary=split_summary,
        epochs=0,
        batch_size=args.batch_size,
        learning_rate=1e-4,
    )
    print(report)
    with open(args.report_path, "w", encoding="utf-8") as handle:
        handle.write(report + "\n")


if __name__ == "__main__":
    main()
