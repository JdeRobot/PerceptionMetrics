from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np
from PIL import Image
import pandas as pd
import pytest

from perceptionmetrics.datasets.cityscapes import (
    CityscapesImageSegmentationDataset,
    build_dataset,
    build_dataset_ontology,
    build_train_id_ontology_translation,
)

_FAKE_ROOT = "/fake/cityscapes"
_FAKE_IMAGE = (
    "/fake/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/"
    "frankfurt/frankfurt_000000_000294_leftImg8bit.png"
)
_FAKE_LABEL = (
    "/fake/cityscapes/gtFine/val/frankfurt/"
    "frankfurt_000000_000294_gtFine_labelIds.png"
)


def _make_patched_build_dataset(
    image_files_by_split: Dict[str, List[str]], label_exists: bool = True
) -> Tuple[dict, dict]:
    """Return a call to build_dataset with filesystem calls mocked.

    :param image_files_by_split: Mapping of split name to list of image file paths
    :type image_files_by_split: dict
    :param label_exists: Whether the expected label path should exist
    :type label_exists: bool
    :return: Result of build_dataset
    :rtype: tuple
    """

    def _fake_glob(pattern: str) -> List[str]:
        for split, files in image_files_by_split.items():
            if f"/{split}/" in pattern:
                return files
        return []

    def _fake_exists(path: str) -> bool:
        return label_exists and path == _FAKE_LABEL

    with patch("perceptionmetrics.datasets.cityscapes.glob", side_effect=_fake_glob):
        with patch(
            "perceptionmetrics.datasets.cityscapes.os.path.exists",
            side_effect=_fake_exists,
        ):
            return build_dataset(val_dataset_root=_FAKE_ROOT)


def test_build_dataset() -> None:
    """Regression tests for Cityscapes build_dataset and ontology handling.

    Verifies that:
    - Cityscapes images are matched with their expected label files.
    - Missing labels are skipped.
    - Raw label IDs are used by default.
    - Train IDs are used when requested.
    - Train-ID mode requires the train-ID label suffix.
    - Ontology translation maps valid classes to themselves.
    """
    dataset, ontology = _make_patched_build_dataset({"val": [_FAKE_IMAGE]})

    assert isinstance(dataset, dict)
    assert "frankfurt_000000_000294" in dataset
    assert dataset["frankfurt_000000_000294"] == (
        _FAKE_IMAGE,
        _FAKE_LABEL,
        "frankfurt",
        "val",
    )
    assert "road" in ontology
    assert ontology["road"]["idx"] == ontology["road"]["cityscapes_id"]

    dataset, _ = _make_patched_build_dataset(
        {"val": [_FAKE_IMAGE]},
        label_exists=False,
    )
    assert dataset == {}

    train_id_ontology = build_dataset_ontology(use_train_id=True)
    assert train_id_ontology["road"]["idx"] == train_id_ontology["road"]["train_id"]
    assert ontology["road"]["idx"] != train_id_ontology["road"]["idx"]

    with pytest.raises(ValueError, match="use_train_id=True requires train-id labels"):
        build_dataset(val_dataset_root=_FAKE_ROOT, use_train_id=True)

    translation = build_train_id_ontology_translation()
    assert translation["road"] == "road"
    assert translation["car"] == "car"


def test_cityscapes_dataset() -> None:
    """Regression tests for the Cityscapes dataset wrapper.

    Verifies that:
    - The wrapper converts the built dataset into a DataFrame.
    - Empty datasets raise a clear error.
    """
    fake_dataset = {
        "frankfurt_000000_000294": (
            _FAKE_IMAGE,
            _FAKE_LABEL,
            "frankfurt",
            "val",
        )
    }
    fake_ontology: Dict[str, Any] = {"road": {"idx": 7, "rgb": [128, 64, 128]}}

    with patch(
        "perceptionmetrics.datasets.cityscapes.build_dataset",
        return_value=(fake_dataset, fake_ontology),
    ):
        dataset = CityscapesImageSegmentationDataset(val_dataset_root=_FAKE_ROOT)

    assert isinstance(dataset.dataset, pd.DataFrame)
    assert len(dataset.dataset) == 1
    assert dataset.dataset.index.tolist() == ["frankfurt_000000_000294"]
    assert dataset.dataset.loc["frankfurt_000000_000294", "scene"] == "frankfurt"
    assert dataset.dataset.loc["frankfurt_000000_000294", "split"] == "val"
    assert dataset.dataset_dir == _FAKE_ROOT

    with patch(
        "perceptionmetrics.datasets.cityscapes.build_dataset",
        return_value=({}, fake_ontology),
    ):
        with pytest.raises(ValueError, match="No Cityscapes samples were found"):
            CityscapesImageSegmentationDataset(val_dataset_root=_FAKE_ROOT)


def test_cityscapes_read_label(tmp_path) -> None:
    """Verify that Cityscapes label-ID images are read as label arrays."""
    label_path = tmp_path / "frankfurt_000000_000294_gtFine_labelIds.png"
    expected = np.array([[7, 8, 11], [12, 13, 17]], dtype=np.uint8)
    Image.fromarray(expected).save(label_path)

    fake_dataset = {
        "frankfurt_000000_000294": (
            _FAKE_IMAGE,
            str(label_path),
            "frankfurt",
            "val",
        )
    }
    fake_ontology: Dict[str, Any] = {"road": {"idx": 7, "rgb": [128, 64, 128]}}

    with patch(
        "perceptionmetrics.datasets.cityscapes.build_dataset",
        return_value=(fake_dataset, fake_ontology),
    ):
        dataset = CityscapesImageSegmentationDataset(val_dataset_root=_FAKE_ROOT)

    label = dataset.read_label(str(label_path))

    assert np.array_equal(label, expected)
