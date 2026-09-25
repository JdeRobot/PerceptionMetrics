from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pandas as pd
import pytest

from perceptionmetrics.datasets.semantickitti import (
    SemanticKITTILiDARSegmentationDataset,
    build_dataset,
    build_ontology,
    build_train_id_ontology_translation,
)

_FAKE_ROOT = "/fake/SemanticKITTI"
_FAKE_CONFIG = "/fake/SemanticKITTI/semantic-kitti.yaml"
_FAKE_POINTS = "/fake/SemanticKITTI/dataset/sequences/08/velodyne/000000.bin"
_FAKE_LABEL = "/fake/SemanticKITTI/dataset/sequences/08/labels/000000.label"
_FAKE_TEST_POINTS = "/fake/SemanticKITTI/dataset/sequences/11/velodyne/000001.bin"

_FAKE_YAML = {
    "labels": {
        0: "unlabeled",
        10: "car",
        252: "moving-car",
    },
    "color_map": {
        0: [0, 0, 0],
        10: [0, 0, 255],
        252: [245, 150, 100],
    },
    "content": {
        0: 0.1,
        10: 0.2,
        252: 0.3,
    },
    "learning_map": {
        0: 0,
        10: 1,
        252: 1,
    },
    "learning_map_inv": {
        0: 0,
        1: 10,
    },
    "split": {
        "valid": [8],
        "test": [11],
    },
}


def _make_patched_build_dataset(
    points_files_by_sequence: Dict[str, List[str]], label_exists: bool = True
) -> Tuple[dict, dict]:
    """Return a call to build_dataset with filesystem calls mocked.

    :param points_files_by_sequence: Mapping of sequence name to point cloud files
    :type points_files_by_sequence: dict
    :param label_exists: Whether non-test labels should exist
    :type label_exists: bool
    :return: Result of build_dataset
    :rtype: tuple
    """

    def _fake_glob(pattern: str, recursive: bool = False) -> List[str]:
        if pattern.endswith("*/velodyne"):
            return [f"{_FAKE_ROOT}/dataset/sequences/{seq}/velodyne" for seq in points_files_by_sequence]
        if pattern.endswith("*/labels"):
            return [f"{_FAKE_ROOT}/dataset/sequences/{seq}/labels" for seq in points_files_by_sequence]
        for sequence, files in points_files_by_sequence.items():
            if f"/{sequence}/velodyne/" in pattern:
                return files
        return []

    def _fake_isdir(path: str) -> bool:
        return path == _FAKE_ROOT or path.endswith("/velodyne")

    def _fake_isfile(path: str) -> bool:
        if path == _FAKE_CONFIG:
            return True
        if path.endswith(".label"):
            return label_exists and path == _FAKE_LABEL
        return path.endswith(".bin")

    with patch(
        "perceptionmetrics.datasets.semantickitti.uio.read_yaml",
        return_value=_FAKE_YAML,
    ):
        with patch(
            "perceptionmetrics.datasets.semantickitti.os.path.isdir",
            side_effect=_fake_isdir,
        ):
            with patch(
                "perceptionmetrics.datasets.semantickitti.os.path.isfile",
                side_effect=_fake_isfile,
            ):
                with patch(
                    "perceptionmetrics.datasets.semantickitti.glob",
                    side_effect=_fake_glob,
                ):
                    return build_dataset(_FAKE_ROOT, _FAKE_CONFIG)


def test_build_dataset() -> None:
    """Regression tests for SemanticKITTI build_dataset and ontology handling.

    Verifies that:
    - validation point clouds are matched with their label files;
    - missing non-test labels are skipped;
    - test samples are kept without labels;
    - raw IDs are used by default;
    - train-ID ontology and translation are built from the YAML config.
    """
    dataset, ontology = _make_patched_build_dataset(
        {"08": [_FAKE_POINTS], "11": [_FAKE_TEST_POINTS]}
    )

    assert dataset["08-000000"] == (_FAKE_POINTS, _FAKE_LABEL, "08", "val")
    assert dataset["11-000001"] == (_FAKE_TEST_POINTS, None, "11", "test")
    assert ontology["car"]["idx"] == 10
    assert ontology["car"]["rgb"] == (255, 0, 0)

    dataset, _ = _make_patched_build_dataset({"08": [_FAKE_POINTS]}, label_exists=False)
    assert dataset == {}

    with patch(
        "perceptionmetrics.datasets.semantickitti.uio.read_yaml",
        return_value=_FAKE_YAML,
    ):
        with patch(
            "perceptionmetrics.datasets.semantickitti.os.path.isfile",
            return_value=True,
        ):
            train_id_ontology = build_ontology(_FAKE_CONFIG, use_train_id=True)
            translation = build_train_id_ontology_translation(_FAKE_CONFIG)

    assert train_id_ontology["car"]["idx"] == 1
    assert translation["moving-car"] == "car"
    assert translation["car"] == "car"


def test_semantic_kitti_dataset() -> None:
    """Regression tests for the SemanticKITTI dataset wrapper.

    Verifies that the wrapper converts the built dataset into a DataFrame.
    """
    fake_dataset = {
        "08-000000": (
            _FAKE_POINTS,
            _FAKE_LABEL,
            "08",
            "val",
        )
    }
    fake_ontology: Dict[str, Any] = {"car": {"idx": 10, "rgb": [255, 0, 0]}}

    with patch(
        "perceptionmetrics.datasets.semantickitti.build_dataset",
        return_value=(fake_dataset, fake_ontology),
    ):
        dataset = SemanticKITTILiDARSegmentationDataset(
            _FAKE_ROOT,
            _FAKE_CONFIG,
            split="val",
        )

    assert isinstance(dataset.dataset, pd.DataFrame)
    assert len(dataset.dataset) == 1
    assert dataset.dataset.index.tolist() == ["08-000000"]
    assert dataset.dataset.loc["08-000000", "scene"] == "08"
    assert dataset.dataset.loc["08-000000", "split"] == "val"
    assert dataset.dataset.attrs["ontology"] == fake_ontology


def test_build_dataset_requires_existing_paths() -> None:
    """Verify that missing SemanticKITTI paths fail early."""
    with pytest.raises(AssertionError, match="Dataset directory not found"):
        build_dataset("/missing/SemanticKITTI", _FAKE_CONFIG)
