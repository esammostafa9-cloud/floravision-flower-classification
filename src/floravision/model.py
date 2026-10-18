"""ResNet-50 transfer-learning classifier, as built in the original notebook."""

from __future__ import annotations

import keras
from keras import layers

from .config import Config


def build_augmentation(cfg: Config) -> keras.Sequential:
    """Train-time augmentation. A flipped, rotated or slightly recoloured rose is still a rose,
    so these create new views without changing the label. Inactive at inference."""
    seed = cfg.seed
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(cfg.aug_rotation, seed=seed),
            layers.RandomBrightness(cfg.aug_brightness, value_range=(0, 255), seed=seed),
            layers.RandomContrast(cfg.aug_contrast, seed=seed),
            layers.RandomZoom(cfg.aug_zoom, seed=seed),
            layers.RandomTranslation(cfg.aug_translation, cfg.aug_translation, seed=seed),
        ],
        name="data_augmentation",
    )


def build_model(num_classes: int, cfg: Config) -> tuple[keras.Model, keras.Model]:
    """Frozen ResNet-50 backbone + small classification head that outputs raw logits.

    The model takes raw 0-255 RGB pixels: augmentation and ResNet-50's ``preprocess_input``
    are part of the graph. Returns ``(model, base_model)`` so the backbone can be partially
    unfrozen for fine-tuning.
    """
    base_model = keras.applications.ResNet50(
        weights=cfg.pretrained_weights,
        include_top=False,
        input_shape=(cfg.img_size, cfg.img_size, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(cfg.img_size, cfg.img_size, 3), name="input_image")
    x = build_augmentation(cfg)(inputs)
    x = keras.applications.resnet50.preprocess_input(x)
    # training=False keeps BatchNorm in inference mode, even after unfreezing.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(cfg.dense_units, activation="relu", name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(cfg.dropout, name="dropout")(x)
    # Raw logits: the loss applies softmax internally, which is numerically stable.
    outputs = layers.Dense(num_classes, name="logits")(x)

    model = keras.Model(inputs, outputs, name="resnet50_flowers")
    return model, base_model


def unfreeze_last_layers(base_model: keras.Model, n_layers: int) -> None:
    """Make the last ``n_layers`` of the backbone trainable.

    BatchNormalization layers stay frozen so their ImageNet statistics are not disturbed by
    the small dataset.
    """
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-n_layers:] if n_layers > 0 else []:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)


def make_loss(cfg: Config) -> keras.losses.Loss:
    return keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=cfg.label_smoothing
    )


def compile_model(model: keras.Model, learning_rate: float, cfg: Config) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=make_loss(cfg),
        metrics=["accuracy"],
    )
