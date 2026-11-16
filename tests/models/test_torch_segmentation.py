import json
from unittest.mock import MagicMock
import warnings

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from PIL import Image
import pytest

# skip these tests when unavailable
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

if isinstance(torch, MagicMock):
    pytest.skip("real torch/torchvision required", allow_module_level=True)

from perceptionmetrics.datasets.segmentation import ImageSegmentationDataset
from perceptionmetrics.models.torch_segmentation import TorchImageSegmentationModel

MODEL_ONTOLOGY = {"void": {"idx": 0}, "grass": {"idx": 1}, "tree": {"idx": 2}}
# Same classes with RELLIS-3D style indices, leaving 1 and 2 unused
NON_CONTIGUOUS_ONTOLOGY = {"void": {"idx": 0}, "grass": {"idx": 3}, "tree": {"idx": 4}}
TO_NON_CONTIGUOUS = np.array([0, 3, 4], dtype=np.uint8)

# GT and predictions in model indices, with a few mistakes
GT = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]], dtype=np.uint8)
PRED = np.array([[0, 1, 1, 1], [0, 0, 1, 2], [2, 2, 2, 2], [0, 2, 2, 2]])

METRIC_ROWS = ["tp", "fp", "fn", "tn", "precision", "recall", "accuracy"]
METRIC_ROWS += ["f1_score", "iou", "dice_score"]
COLUMNS = ["void", "grass", "tree", "macro", "micro"]


class FixedSegmenter(torch.nn.Module):
    """Model whose logits always predict PRED, in model indices."""

    def __init__(self):
        super().__init__()
        logits = torch.nn.functional.one_hot(torch.tensor(PRED), 3)
        self.register_buffer("logits", logits.permute(2, 0, 1)[None].float())

    def forward(self, x):
        """Return the stored logits for every image in the batch.

        :param x: Input image batch
        :type x: torch.Tensor
        :return: Logits with shape (batch, classes, height, width)
        :rtype: torch.Tensor
        """
        return self.logits.expand(x.shape[0], -1, -1, -1)


def evaluate(root, dataset_ontology, gt, direction):
    """Evaluate FixedSegmenter on a one-image dataset.

    :param root: Directory for the dataset and model files
    :type root: pathlib.Path
    :param dataset_ontology: Dataset ontology
    :type dataset_ontology: dict
    :param gt: Ground truth label in dataset indices
    :type gt: np.ndarray
    :param direction: Ontology translation direction
    :type direction: str
    :return: Evaluation results
    :rtype: pd.DataFrame
    """
    root.mkdir()
    Image.new("RGB", (4, 4)).save(root / "image.png")
    Image.fromarray(gt).save(root / "label.png")
    (root / "ontology.json").write_text(json.dumps(MODEL_ONTOLOGY))
    (root / "cfg.json").write_text(json.dumps({"num_workers": 0}))

    dataset = ImageSegmentationDataset(
        pd.DataFrame(
            {"image": ["image.png"], "label": ["label.png"], "split": ["test"]},
            index=["sample"],
        ),
        str(root),
        dataset_ontology,
    )
    model = TorchImageSegmentationModel(
        FixedSegmenter(), str(root / "cfg.json"), str(root / "ontology.json")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # differing indices warning is expected
        return model.eval(dataset, split="test", translation_direction=direction)


# To verify that both translation directions match a contiguous-ontology baseline
@pytest.mark.parametrize("direction", ["dataset_to_model", "model_to_dataset"])
def test_eval_non_contiguous_dataset_indices(tmp_path, direction):
    """Evaluate against a non-contiguous dataset ontology in both directions.

    :param tmp_path: Temporary directory provided by pytest
    :type tmp_path: pathlib.Path
    :param direction: Ontology translation direction
    :type direction: str
    """
    baseline = evaluate(tmp_path / "baseline", MODEL_ONTOLOGY, GT, direction)
    results = evaluate(
        tmp_path / "non_contiguous",
        NON_CONTIGUOUS_ONTOLOGY,
        TO_NON_CONTIGUOUS[GT],
        direction,
    )

    assert_frame_equal(
        results.loc[METRIC_ROWS, COLUMNS], baseline.loc[METRIC_ROWS, COLUMNS]
    )
