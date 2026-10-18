"""Train -> evaluate -> predict on the tiny synthetic dataset, through the real CLIs."""

import json

import pandas as pd
import pytest

from floravision import evaluate, predict, train


@pytest.mark.slow
def test_train_evaluate_predict(tiny_dataset, tiny_config, tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    tiny_config.to_yaml(cfg_path)
    run = tmp_path / "run"

    train.main(
        ["--data-dir", str(tiny_dataset), "--config", str(cfg_path), "--output-dir", str(run)]
    )

    for name in [
        "model.keras",
        "config.yaml",
        "class_names.json",
        "history.json",
        "selection.json",
        "test_metrics.json",
        "classification_report.csv",
        "test_predictions.csv",
        "figures/training_curves.png",
        "figures/confusion_matrix.png",
        "figures/confident_mistakes.png",
        "figures/sample_predictions.png",
    ]:
        assert (run / name).exists(), name

    history = json.loads((run / "history.json").read_text())
    assert set(history) == {"phase1_head", "phase2_finetune"}
    assert len(history["phase1_head"]["gradient_norm"]) == 1

    metrics = json.loads((run / "test_metrics.json").read_text())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["test_images"] == 6  # 20% of 30 images

    # Standalone evaluation picks up config.yaml and class_names.json from the run folder.
    eval_dir = tmp_path / "eval"
    evaluate.main(
        [
            "--model",
            str(run / "model.keras"),
            "--data-dir",
            str(tiny_dataset),
            "--output-dir",
            str(eval_dir),
        ]
    )
    assert json.loads((eval_dir / "test_metrics.json").read_text()) == metrics

    out_csv = tmp_path / "pred.csv"
    predict.main(
        [
            "--model",
            str(run / "model.keras"),
            str(tiny_dataset / "roses"),
            "--top-k",
            "2",
            "--output",
            str(out_csv),
        ]
    )
    preds = pd.read_csv(out_csv)
    assert len(preds) == 10  # the .txt file in the folder is ignored
    assert set(preds.columns) >= {"image", "top1", "top1_prob", "top2", "top2_prob"}
