"""Model architecture and training callbacks for BraTS 2020 segmentation."""

from __future__ import annotations

import os
from typing import List, Tuple

os.environ.setdefault("SM_FRAMEWORK", "tf.keras")


def build_unet_efficientnetb0(
    input_shape: Tuple[int, int, int] = (256, 256, 4),
    classes: int = 3,
    learning_rate: float = 1e-4,
    encoder_weights: str | None = None,
):
    """Build and compile a U-Net with EfficientNetB0 backbone for 4-channel MRI.

    Uses encoder_weights=None by default because 4-channel MRI (T1, T1c, T2, FLAIR)
    does not match 3-channel RGB ImageNet weights without input projection.
    """
    import segmentation_models as sm
    import tensorflow as tf

    model = sm.Unet(
        "efficientnetb0",
        input_shape=input_shape,
        classes=classes,
        activation="sigmoid",
        encoder_weights=encoder_weights,
    )

    total_loss = sm.losses.DiceLoss() + sm.losses.BinaryFocalLoss()
    metrics = [sm.metrics.iou_score, sm.metrics.f1_score]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=total_loss,
        metrics=metrics,
    )
    return model


def build_callbacks(checkpoint_path: str = "best_model_weights.h5") -> List[object]:
    """Return ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau callbacks."""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    return [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_iou_score",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
