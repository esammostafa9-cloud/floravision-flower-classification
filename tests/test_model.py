import numpy as np
import pytest
from keras import layers

from floravision.config import Config
from floravision.model import build_model, make_loss, unfreeze_last_layers


def test_model_outputs_logits_for_raw_pixels(tiny_config):
    model, base = build_model(num_classes=5, cfg=tiny_config)
    assert not base.trainable
    assert model.layers[-1].name == "logits"

    out = model.predict(np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32"), verbose=0)
    assert out.shape == (2, 5)


def test_preprocess_input_is_inside_the_model(tiny_config):
    """ResNet-50 normalisation lives in the graph, so callers must pass raw 0-255 pixels."""
    model, _ = build_model(num_classes=3, cfg=tiny_config)
    raw = np.full((1, 32, 32, 3), 200.0, dtype="float32")
    pre = raw - np.array([103.939, 116.779, 123.68], dtype="float32")  # approx. preprocessed
    assert not np.allclose(
        model.predict(raw, verbose=0), model.predict(pre, verbose=0), atol=1e-4
    ), "the model ignores its own preprocessing"


def test_unfreeze_last_layers_keeps_batchnorm_frozen(tiny_config):
    _, base = build_model(num_classes=5, cfg=tiny_config)
    unfreeze_last_layers(base, 15)

    trainable = [layer for layer in base.layers if layer.trainable]
    assert trainable, "expected some layers to be unfrozen"
    assert all(layer in base.layers[-15:] for layer in trainable)
    assert not any(isinstance(layer, layers.BatchNormalization) for layer in trainable)
    assert not any(layer.trainable for layer in base.layers[:-15])


def test_unfreeze_zero_layers_freezes_everything(tiny_config):
    _, base = build_model(num_classes=5, cfg=tiny_config)
    unfreeze_last_layers(base, 0)
    assert not any(layer.trainable for layer in base.layers)


def test_loss_uses_label_smoothing(tiny_config):
    loss = make_loss(tiny_config)
    perfect = loss(np.array([[1.0, 0.0]]), np.array([[50.0, -50.0]]))
    assert float(perfect) > 0.01  # smoothing keeps a floor even for a perfect prediction


def test_config_yaml_round_trip(tmp_path):
    cfg = Config(head_epochs=3, fine_tune_layers=10)
    cfg.to_yaml(tmp_path / "c.yaml")
    assert Config.from_yaml(tmp_path / "c.yaml") == cfg
    assert Config.from_yaml(tmp_path / "c.yaml", head_epochs=7, batch_size=None).head_epochs == 7


def test_config_rejects_unknown_keys_and_bad_values(tmp_path):
    (tmp_path / "bad.yaml").write_text("learning_rate: 0.1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_yaml(tmp_path / "bad.yaml")
    with pytest.raises(ValueError):
        Config(val_size=0.6, test_size=0.5)


def test_config_yaml_is_loaded_safely(tmp_path):
    """yaml.safe_load must refuse Python object tags."""
    (tmp_path / "evil.yaml").write_text("seed: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(Exception, match="(?i)constructor|could not determine"):
        Config.from_yaml(tmp_path / "evil.yaml")
