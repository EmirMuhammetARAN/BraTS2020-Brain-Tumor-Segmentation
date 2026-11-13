"""CLI entrypoint for training and evaluating the BraTS 2020 segmentation model."""

from __future__ import annotations

import argparse

from src.data import build_data_generator, discover_h5_slices, split_by_patient
from src.evaluate import evaluate_generator, plot_confusion_matrices, plot_predictions_heatmap
from src.metrics import format_evaluation_report
from src.model import build_callbacks, build_unet_efficientnetb0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train U-Net + EfficientNetB0 on BraTS 2020 with patient-level splits."
    )
    parser.add_argument("--data-dir", default="BraTS2020_training_data/content/data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights-out", default="best_model_weights.h5")
    parser.add_argument("--report-out", default="evaluation_report.txt")
    args = parser.parse_args()

    all_files = discover_h5_slices(args.data_dir)
    train_files, val_files, split_summary = split_by_patient(
        all_files, test_size=args.test_size, random_state=args.seed
    )
    print(f"Patient-level split summary: {split_summary}")

    train_gen = build_data_generator(
        train_files, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_gen = build_data_generator(val_files, batch_size=args.batch_size, shuffle=False)

    model = build_unet_efficientnetb0(learning_rate=args.lr, encoder_weights=None)
    callbacks = build_callbacks(checkpoint_path=args.weights_out)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.load_weights(args.weights_out)
    metrics = evaluate_generator(model, val_gen, max_batches=None)
    plot_confusion_matrices(metrics)
    plot_predictions_heatmap(model, val_gen)

    report = format_evaluation_report(
        metrics=metrics,
        total_params=model.count_params(),
        split_summary=split_summary,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print(report)
    with open(args.report_out, "w", encoding="utf-8") as handle:
        handle.write(report + "\n")


if __name__ == "__main__":
    main()
