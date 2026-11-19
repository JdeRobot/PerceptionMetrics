import json
import os

import numpy as np
import pandas as pd
import pytest

from perceptionmetrics.datasets.detection import ImageDetectionDataset

ONTOLOGY = {"car": {"idx": 0}, "person": {"idx": 1}}


class _JsonDetectionDataset(ImageDetectionDataset):
    """Minimal detection dataset whose GT annotations are small JSON files."""

    def read_annotation(self, fname):
        with open(fname) as f:
            ann = json.load(f)
        return ann["boxes"], ann["labels"]


def _make_dataset(tmp_path):
    ann_file = tmp_path / "sample0_gt.json"
    ann_file.write_text(
        json.dumps({"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "labels": [0, 1]})
    )
    df = pd.DataFrame(
        {"image": [None], "annotation": [str(ann_file)], "split": ["test"]},
        index=["sample0"],
    )
    return _JsonDetectionDataset(df, str(tmp_path), ONTOLOGY)


def _write_preds(preds_dir, labels):
    os.makedirs(preds_dir, exist_ok=True)
    # Same record layout as TorchImageDetectionModel.eval(results_per_sample=True)
    preds = pd.DataFrame(
        [
            {
                "image_id": "sample0",
                "label": labels[0],
                "score": 0.9,
                "bbox": [0, 0, 10, 10],
            },
            {
                "image_id": "sample0",
                "label": labels[1],
                "score": 0.8,
                "bbox": [20, 20, 30, 30],
            },
        ]
    )
    preds.to_json(os.path.join(preds_dir, "sample0.json"), orient="records", indent=2)


def test_eval_preds_accepts_class_names_saved_by_model_eval(tmp_path):
    """Predictions saved with class names (as pm_eval_model writes them) can be evaluated (#682)."""
    dataset = _make_dataset(tmp_path)
    preds_dir = tmp_path / "preds"
    _write_preds(preds_dir, ["car", "person"])

    results = dataset.eval_preds(str(preds_dir), split="test")

    assert np.allclose(results.loc["AP", ["car", "person"]].astype(float), 1.0)


def test_eval_preds_still_accepts_integer_labels(tmp_path):
    dataset = _make_dataset(tmp_path)
    preds_dir = tmp_path / "preds"
    _write_preds(preds_dir, [0, 1])

    results = dataset.eval_preds(str(preds_dir), split="test")

    assert np.allclose(results.loc["AP", ["car", "person"]].astype(float), 1.0)


def test_eval_preds_rejects_unknown_class_name(tmp_path):
    dataset = _make_dataset(tmp_path)
    preds_dir = tmp_path / "preds"
    _write_preds(preds_dir, ["car", "truck"])

    with pytest.raises(ValueError, match="Unknown class name 'truck'"):
        dataset.eval_preds(str(preds_dir), split="test")
