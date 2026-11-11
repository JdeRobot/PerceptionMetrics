import json
import os
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest

# skip these tests when unavailable
tf = pytest.importorskip("tensorflow")

if isinstance(tf, MagicMock):
    pytest.skip("real tensorflow required", allow_module_level=True)

from perceptionmetrics.datasets.generic import GenericImageSegmentationDataset
from perceptionmetrics.models.tf_segmentation import TensorflowImageSegmentationModel
import perceptionmetrics.utils.segmentation_metrics as um

HEIGHT = WIDTH = 8


def _build_eval_setup(root, n_samples, batch_size):
    """Create a tiny test split, a Keras model and the model configuration files"""
    data_dir = root / "test"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_samples):
        rgb = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(data_dir / f"s{i}_data.png")
        label = rng.integers(0, 3, (HEIGHT, WIDTH), dtype=np.uint8)
        Image.fromarray(label).save(data_dir / f"s{i}_label.png")

    ontology = {name: {"idx": i, "rgb": [i, i, i]} for i, name in enumerate("abc")}
    ontology_fname = root / "ontology.json"
    ontology_fname.write_text(json.dumps(ontology))
    cfg_fname = root / "cfg.json"
    cfg_fname.write_text(json.dumps({"batch_size": batch_size}))

    dataset = GenericImageSegmentationDataset(
        "*_data.png",
        "*_label.png",
        str(ontology_fname),
        test_dataset_dir=str(data_dir),
    )
    inputs = tf.keras.Input((None, None, 3))
    keras_model = tf.keras.Model(inputs, tf.keras.layers.Conv2D(3, 1)(inputs))
    model = TensorflowImageSegmentationModel(
        keras_model, str(cfg_fname), str(ontology_fname)
    )
    return model, dataset


# To verify that every sample reaches the metrics even if the last batch is partial
@pytest.mark.parametrize("n_samples, batch_size", [(5, 2), (3, 5), (4, 2)])
def test_eval_keeps_partial_last_batch(tmp_path, monkeypatch, n_samples, batch_size):
    model, dataset = _build_eval_setup(tmp_path, n_samples, batch_size)

    factories = []

    class SpyFactory(um.SegmentationMetricsFactory):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            factories.append(self)

    monkeypatch.setattr(um, "SegmentationMetricsFactory", SpyFactory)

    predictions_outdir = tmp_path / "preds"
    model.eval(dataset, split="test", predictions_outdir=str(predictions_outdir))

    assert factories[0].confusion_matrix.sum() == n_samples * HEIGHT * WIDTH
    assert len(os.listdir(predictions_outdir)) == n_samples
