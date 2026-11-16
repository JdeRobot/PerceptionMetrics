import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from PIL import Image
import pytest

from perceptionmetrics.datasets.segmentation import ImageSegmentationDataset

# Same classes as a contiguous baseline, but with unused indices 1 and 2
CONTIGUOUS_ONTOLOGY = {"a": {"idx": 0}, "b": {"idx": 1}, "c": {"idx": 2}}
NON_CONTIGUOUS_ONTOLOGY = {"a": {"idx": 0}, "b": {"idx": 3}, "c": {"idx": 4}}
TO_NON_CONTIGUOUS = np.array([0, 3, 4], dtype=np.uint8)

# GT and predictions in contiguous indices, with a few mistakes
GT = np.array([[0, 0, 0, 1], [1, 2, 2, 2], [2, 2, 0, 0], [1, 1, 2, 0]], dtype=np.uint8)
PRED = np.array(
    [[0, 0, 1, 1], [2, 2, 2, 2], [0, 2, 0, 0], [1, 2, 2, 0]], dtype=np.uint8
)

METRIC_ROWS = ["tp", "fp", "fn", "tn", "precision", "recall", "accuracy"]
METRIC_ROWS += ["f1_score", "iou", "dice_score"]
COLUMNS = ["a", "b", "c", "macro", "micro"]


def build_dataset(root, ontology, gt, pred):
    """Write one GT label and its prediction, and build the dataset.

    :param root: Directory for the dataset and the predictions
    :type root: pathlib.Path
    :param ontology: Dataset ontology
    :type ontology: dict
    :param gt: Ground truth label
    :type gt: np.ndarray
    :param pred: Predicted label
    :type pred: np.ndarray
    :return: Dataset and predictions directory
    :rtype: Tuple[ImageSegmentationDataset, str]
    """
    (root / "preds").mkdir(parents=True)
    Image.new("RGB", gt.shape[::-1]).save(root / "image.png")
    Image.fromarray(gt).save(root / "label.png")
    Image.fromarray(pred).save(root / "preds" / "label.png")
    dataset = ImageSegmentationDataset(
        pd.DataFrame(
            {"image": ["image.png"], "label": ["label.png"], "split": ["test"]},
            index=["sample"],
        ),
        str(root),
        ontology,
    )
    return dataset, str(root / "preds")


def eval_contiguous_baseline(root):
    """Evaluate the contiguous baseline.

    :param root: Directory for the baseline dataset
    :type root: pathlib.Path
    :return: Metrics for the baseline
    :rtype: pd.DataFrame
    """
    dataset, preds_dir = build_dataset(root, CONTIGUOUS_ONTOLOGY, GT, PRED)
    return dataset.eval_preds(preds_dir, split="test")


# To verify that predictions in the same non-contiguous ontology match the baseline
def test_eval_preds_non_contiguous_ontology(tmp_path):
    """Evaluate GT and predictions that share a non-contiguous ontology.

    :param tmp_path: Temporary directory provided by pytest
    :type tmp_path: pathlib.Path
    """
    baseline = eval_contiguous_baseline(tmp_path / "baseline")
    dataset, preds_dir = build_dataset(
        tmp_path / "non_contiguous",
        NON_CONTIGUOUS_ONTOLOGY,
        TO_NON_CONTIGUOUS[GT],
        TO_NON_CONTIGUOUS[PRED],
    )

    results = dataset.eval_preds(preds_dir, split="test")

    assert_frame_equal(
        results.loc[METRIC_ROWS, COLUMNS], baseline.loc[METRIC_ROWS, COLUMNS]
    )


# To verify that both translation directions match the baseline
@pytest.mark.parametrize("direction", ["dataset_to_model", "model_to_dataset"])
def test_eval_preds_non_contiguous_ontology_translation(tmp_path, direction):
    """Evaluate contiguous predictions against non-contiguous GT in both directions.

    :param tmp_path: Temporary directory provided by pytest
    :type tmp_path: pathlib.Path
    :param direction: Ontology translation direction
    :type direction: str
    """
    baseline = eval_contiguous_baseline(tmp_path / "baseline")
    dataset, preds_dir = build_dataset(
        tmp_path / "non_contiguous",
        NON_CONTIGUOUS_ONTOLOGY,
        TO_NON_CONTIGUOUS[GT],
        PRED,
    )

    results = dataset.eval_preds(
        preds_dir,
        split="test",
        ontology_translation={"a": "a", "b": "b", "c": "c"},
        translation_direction=direction,
        pred_ontology=CONTIGUOUS_ONTOLOGY,
    )

    assert_frame_equal(
        results.loc[METRIC_ROWS, COLUMNS], baseline.loc[METRIC_ROWS, COLUMNS]
    )
