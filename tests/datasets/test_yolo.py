import logging
import os
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pandas as pd
import pytest
from PIL import Image
from perceptionmetrics.datasets.yolo import YOLODataset, build_dataset  # noqa: E402

_FAKE_YAML_TRAIN_VAL_ONLY = {
    "path": "/fake/dataset",
    "train": "images/train",
    "val": "images/val",
    # 'test' key is absent — common for many YOLO datasets
    "names": {0: "cat", 1: "dog"},
}

_FAKE_YAML_TEST_NULL = {
    "path": "/fake/dataset",
    "train": "images/train",
    "val": "images/val",
    "test": None,  # Key present but value is null (as parsed from YAML)
    "names": {0: "cat", 1: "dog"},
}

_FAKE_YAML_ALL_SPLITS = {
    "path": "/fake/dataset",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "cat", 1: "dog"},
}


def _make_patched_build_dataset(
    yaml_content: Dict[str, Any], label_files_by_split: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """Return a call to build_dataset with filesystem calls mocked.

    :param yaml_content: Dictionary simulating parsed YAML content
    :type yaml_content: dict
    :param label_files_by_split: Mapping of split name to list of label file paths
    :type label_files_by_split: dict
    :return: Result of build_dataset
    :rtype: tuple
    """
    fake_dataset_fname = "/fake/dataset/data.yaml"
    fake_dataset_dir = "/fake/dataset"

    def _fake_glob(pattern: str) -> List[str]:
        for split, files in label_files_by_split.items():
            if split in pattern:
                return files
        return []

    def _fake_isfile(path: str) -> bool:
        # Treat any .txt file as label, any other as image
        return True

    with patch("os.path.isfile", return_value=True), patch(
        "os.path.isdir", return_value=True
    ), patch(
        "perceptionmetrics.datasets.yolo.uio.read_yaml", return_value=yaml_content
    ), patch(
        "perceptionmetrics.datasets.yolo.glob", side_effect=_fake_glob
    ):
        return build_dataset(fake_dataset_fname, fake_dataset_dir)


def test_build_dataset(caplog: pytest.LogCaptureFixture) -> None:
    """Regression tests for build_dataset and split validation.

    Verifies that:
    - No TypeError is raised when 'test' key is absent from the YAML.
    - No TypeError is raised when 'test' key is present but null (e.g. COCO8).
    - A logging.WARNING is emitted for each missing or null split.
    - All three splits are loaded correctly when all paths are defined.
    - The ontology is built correctly from the YAML 'names' dict.

    :param caplog: pytest log capture fixture
    :type caplog: pytest.LogCaptureFixture
    """
    # --- missing 'test' key: must not raise TypeError, must skip 'test' rows ---
    dataset, ontology, _ = _make_patched_build_dataset(
        _FAKE_YAML_TRAIN_VAL_ONLY,
        {
            "train": ["/fake/dataset/labels/train/img1.txt"],
            "val": ["/fake/dataset/labels/val/img2.txt"],
        },
    )
    assert isinstance(dataset, pd.DataFrame)
    assert "test" not in dataset["split"].values
    assert set(dataset["split"].unique()) <= {"train", "val"}

    # --- 'test: null' (exact COCO8 structure): must not raise TypeError ---
    dataset, _, _ = _make_patched_build_dataset(
        _FAKE_YAML_TEST_NULL,
        {
            "train": ["/fake/dataset/labels/train/img1.txt"],
            "val": ["/fake/dataset/labels/val/img2.txt"],
        },
    )
    assert "test" not in dataset["split"].values

    # --- warning must be emitted for the null 'test' split ---
    with caplog.at_level(logging.WARNING, logger="root"):
        _make_patched_build_dataset(
            _FAKE_YAML_TEST_NULL,
            {
                "train": ["/fake/dataset/labels/train/img1.txt"],
                "val": ["/fake/dataset/labels/val/img2.txt"],
            },
        )
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("test" in msg for msg in warning_messages), (
        "Expected a warning about the missing 'test' split, got: %s" % warning_messages
    )

    # --- all splits defined: all three must appear in the DataFrame ---
    dataset, _, _ = _make_patched_build_dataset(
        _FAKE_YAML_ALL_SPLITS,
        {
            "train": ["/fake/dataset/labels/train/img1.txt"],
            "val": ["/fake/dataset/labels/val/img2.txt"],
            "test": ["/fake/dataset/labels/test/img3.txt"],
        },
    )
    assert set(dataset["split"].unique()) == {"train", "val", "test"}

    # --- ontology must be built correctly from the YAML 'names' dict ---
    _, ontology, _ = _make_patched_build_dataset(
        _FAKE_YAML_TRAIN_VAL_ONLY,
        {"train": ["/fake/dataset/labels/train/img1.txt"]},
    )
    assert "cat" in ontology and "dog" in ontology
    assert ontology["cat"]["idx"] == 0
    assert ontology["dog"]["idx"] == 1


def test_build_dataset_keeps_samples_with_dots_in_name() -> None:
    """Samples whose file names contain extra dots must not collapse into one.

    Roboflow exports name every augmented copy of an image
    ``<original>_jpg.rf.<hash>.jpg``, so the part before the first dot is the
    same for all of them.
    """
    labels = [
        "/fake/dataset/labels/val/IMG_0001_jpg.rf.3f9a1c7e2b.txt",
        "/fake/dataset/labels/val/IMG_0001_jpg.rf.8b21d04aa9.txt",
        "/fake/dataset/labels/val/IMG_0002_jpg.rf.1a2b3c4d5e.txt",
    ]
    dataset, _, _ = _make_patched_build_dataset(
        _FAKE_YAML_TRAIN_VAL_ONLY, {"val": labels}
    )

    assert len(dataset) == 3
    assert "IMG_0001_jpg.rf.3f9a1c7e2b" in dataset.index
    assert "IMG_0001_jpg.rf.8b21d04aa9" in dataset.index

    # Plain file names keep the same sample name as before
    dataset, _, _ = _make_patched_build_dataset(
        _FAKE_YAML_TRAIN_VAL_ONLY, {"val": ["/fake/dataset/labels/val/img1.txt"]}
    )
    assert list(dataset.index) == ["img1"]


def _write_yolo_dataset(dataset_dir) -> str:
    """Write a small YOLO dataset to disk and return its YAML path.

    :param dataset_dir: Directory where the dataset is created
    :type dataset_dir: pathlib.Path
    :return: Path to the dataset YAML file
    :rtype: str
    """
    for split, sample in [("train", "img1"), ("val", "img2")]:
        (dataset_dir / "images" / split).mkdir(parents=True)
        (dataset_dir / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (100, 50)).save(
            dataset_dir / "images" / split / f"{sample}.jpg"
        )
        (dataset_dir / "labels" / split / f"{sample}.txt").write_text(
            "0 0.5 0.5 0.2 0.4\n"
        )

    yaml_fname = dataset_dir / "data.yaml"
    yaml_fname.write_text(
        f"path: {dataset_dir}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: cat\n"
    )
    return str(yaml_fname)


@pytest.mark.parametrize("parent", ["drone_images", "my_labels"])
def test_yolo_dataset_parent_folder_names(tmp_path, parent) -> None:
    """Only the dataset's own images/labels folders must be swapped.

    A parent folder whose name contains "images" or "labels" must not be
    rewritten when looking up the matching labels or image file.
    """
    dataset_dir = tmp_path / parent / "mydata"
    yaml_fname = _write_yolo_dataset(dataset_dir)

    dataset = YOLODataset(yaml_fname, None)
    assert sorted(dataset.dataset.index) == ["img1", "img2"]
    assert dataset.dataset.loc["img2", "annotation"] == os.path.join(
        "labels", "val", "img2.txt"
    )

    ann_fname = os.path.join(
        dataset.dataset_dir, dataset.dataset.loc["img2", "annotation"]
    )
    boxes, labels = dataset.read_annotation(ann_fname)
    assert boxes == [[40.0, 15.0, 60.0, 35.0]]
    assert labels == [0]
