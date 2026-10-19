import json
from unittest.mock import MagicMock
import warnings

import numpy as np
import pandas as pd
from PIL import Image
import pytest

# skip these tests when unavailable
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

if isinstance(torch, MagicMock):
    pytest.skip("real torch/torchvision required", allow_module_level=True)

from perceptionmetrics.datasets.segmentation import ImageSegmentationDataset
from perceptionmetrics.models.torch_segmentation import TorchImageSegmentationModel

# Ground truth expressed in model indices (void, grass, tree)
GT_MODEL_IDX = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])


class PerfectSegmenter(torch.nn.Module):
    """Model whose logits always match the ground truth in model indices."""

    def __init__(self):
        super().__init__()
        logits = torch.nn.functional.one_hot(torch.tensor(GT_MODEL_IDX), 3)
        self.register_buffer("logits", logits.permute(2, 0, 1)[None].float())

    def forward(self, x):
        """Return the stored logits for every image in the batch.

        :param x: Input image batch
        :type x: torch.Tensor
        :return: Logits with shape (batch, classes, height, width)
        :rtype: torch.Tensor
        """
        return self.logits.expand(x.shape[0], -1, -1, -1)


def build_model_and_dataset(tmp_path):
    """Build a one-sample dataset with non-contiguous indices and a perfect model.

    :param tmp_path: Temporary directory provided by pytest
    :type tmp_path: pathlib.Path
    :return: Model and dataset ready for evaluation
    :rtype: Tuple[TorchImageSegmentationModel, ImageSegmentationDataset]
    """
    dataset_ontology = {"void": {"idx": 0}, "grass": {"idx": 3}, "tree": {"idx": 4}}
    model_ontology = {"void": {"idx": 0}, "grass": {"idx": 1}, "tree": {"idx": 2}}

    gt = np.array([0, 3, 4], dtype=np.uint8)[GT_MODEL_IDX]
    Image.new("RGB", (4, 4)).save(tmp_path / "image.png")
    Image.fromarray(gt).save(tmp_path / "label.png")
    (tmp_path / "ontology.json").write_text(json.dumps(model_ontology))
    (tmp_path / "cfg.json").write_text(json.dumps({"num_workers": 0}))

    dataset = ImageSegmentationDataset(
        pd.DataFrame(
            {"image": ["image.png"], "label": ["label.png"], "split": ["test"]},
            index=["sample"],
        ),
        str(tmp_path),
        dataset_ontology,
    )
    model = TorchImageSegmentationModel(
        PerfectSegmenter(),
        str(tmp_path / "cfg.json"),
        str(tmp_path / "ontology.json"),
    )
    return model, dataset


# To verify that both translation directions work with non-contiguous dataset indices
@pytest.mark.parametrize("direction", ["dataset_to_model", "model_to_dataset"])
def test_eval_non_contiguous_dataset_indices(tmp_path, direction):
    """Evaluate a perfect model in both translation directions.

    :param tmp_path: Temporary directory provided by pytest
    :type tmp_path: pathlib.Path
    :param direction: Ontology translation direction
    :type direction: str
    """
    model, dataset = build_model_and_dataset(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # differing indices warning is expected
        results = model.eval(dataset, split="test", translation_direction=direction)

    assert results.loc["iou", ["void", "grass", "tree"]].tolist() == [1.0, 1.0, 1.0]
    assert results.loc["iou", "macro"] == 1.0
