import json
import logging
import sys
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

# Stub out heavy optional C-extensions that are not available in the test
# environment (open3d, supervision, etc.) before importing any perceptionmetrics module.
for _stub in ("open3d", "supervision"):
    if _stub not in sys.modules:
        sys.modules[_stub] = MagicMock()

# Ensure tqdm.tqdm is a callable no-op (used in detection.py)
if "tqdm" not in sys.modules:
    import tqdm
_tqdm_mod = sys.modules["tqdm"]
if not callable(getattr(_tqdm_mod, "tqdm", None)):
    _tqdm_mod.tqdm = lambda iterable, **kw: iterable  # type: ignore[attr-defined]

from perceptionmetrics.datasets.bdd100k import (  # noqa: E402
    BDD100KDetectionDataset,
    build_bdd100k_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_ANNOTATION_VALID = {
    "name": "img1.jpg",
    "frames": [
        {
            "objects": [
                {
                    "category": "car",
                    "box2d": {"x1": 10, "y1": 20, "x2": 100, "y2": 80},
                },
                {
                    "category": "person",
                    "box2d": {"x1": 50, "y1": 60, "x2": 150, "y2": 200},
                },
            ]
        }
    ],
}

_FAKE_ANNOTATION_NO_BOX2D = {
    "name": "img2.jpg",
    "frames": [
        {
            "objects": [
                {
                    "category": "car",
                    "box2d": {"x1": 10, "y1": 20, "x2": 100, "y2": 80},
                },
                {
                    "category": "lane",
                    "poly2d": [{"vertices": [[0, 0], [1, 1]]}],
                },
            ]
        }
    ],
}

_FAKE_ANNOTATION_UNKNOWN_CATEGORY = {
    "name": "img3.jpg",
    "frames": [
        {
            "objects": [
                {
                    "category": "car",
                    "box2d": {"x1": 5, "y1": 10, "x2": 50, "y2": 40},
                },
                {
                    "category": "helicopter",
                    "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                },
            ]
        }
    ],
}


def _make_patched_dataset(annotation):
    """Return a BDD100KDetectionDataset with filesystem calls mocked.

    The dataset is created with empty glob results (no images discovered),
    and ``read_annotation`` is invoked with the given annotation dict mocked
    into ``builtins.open`` / ``json.load``.

    :param annotation: Dictionary simulating a BDD100K per-image JSON
    :type annotation: dict
    :return: Tuple of (dataset instance, boxes, category_indices)
    :rtype: tuple
    """

    def _fake_glob(pattern):
        return []

    with patch(
        "perceptionmetrics.datasets.bdd100k.glob", side_effect=_fake_glob
    ), patch("os.path.isdir", return_value=True):
        ds = BDD100KDetectionDataset("/images", "/labels")

    with patch("builtins.open", mock_open(read_data=json.dumps(annotation))), patch(
        "json.load", return_value=annotation
    ):
        boxes, cat_indices = ds.read_annotation("/labels/train/fake.json")

    return ds, boxes, cat_indices


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_bdd100k_dataset():
    """Verify build_bdd100k_dataset pairs train/val images with label paths.

    Checks that the returned DataFrame has the correct columns, rows, and
    split values for discovered images that have matching label files.
    """

    def _fake_glob(pattern):
        if "train" in pattern and pattern.endswith("*.jpg"):
            return ["/images/train/img1.jpg", "/images/train/img2.jpg"]
        if "val" in pattern and pattern.endswith("*.jpg"):
            return ["/images/val/img3.jpg"]
        return []

    with patch(
        "perceptionmetrics.datasets.bdd100k.glob", side_effect=_fake_glob
    ), patch("os.path.isdir", return_value=True), patch(
        "os.path.isfile", return_value=True
    ):
        dataset, ontology = build_bdd100k_dataset("/images", "/labels")

    assert isinstance(dataset, pd.DataFrame)
    assert list(dataset.columns) == ["image", "annotation", "split"]
    assert len(dataset) == 3
    assert set(dataset["split"].unique()) == {"train", "val"}

    # --- ontology must contain all 10 BDD100K classes ---
    assert len(ontology) == 10
    assert ontology["car"]["idx"] == 0
    assert ontology["train"]["idx"] == 9


def test_missing_label_skipped(caplog):
    """Images with no matching label file are skipped with a warning.

    :param caplog: pytest log capture fixture
    :type caplog: pytest.LogCaptureFixture
    """

    def _fake_glob(pattern):
        if "train" in pattern and pattern.endswith("*.jpg"):
            return ["/images/train/img1.jpg"]
        return []

    def _fake_isfile(path):
        # Label file does not exist
        if path.endswith(".json"):
            return False
        return True

    with patch(
        "perceptionmetrics.datasets.bdd100k.glob", side_effect=_fake_glob
    ), patch("os.path.isdir", return_value=True), patch(
        "os.path.isfile", side_effect=_fake_isfile
    ):
        with caplog.at_level(logging.WARNING, logger="root"):
            dataset, _ = build_bdd100k_dataset("/images", "/labels")

    assert len(dataset) == 0
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("No matching label file" in msg for msg in warning_messages)


def test_read_annotation_correct():
    """read_annotation returns correct boxes and category indices for known categories."""
    _, boxes, cat_indices = _make_patched_dataset(_FAKE_ANNOTATION_VALID)

    assert len(boxes) == 2
    assert boxes[0] == [10, 20, 100, 80]
    assert boxes[1] == [50, 60, 150, 200]
    assert cat_indices[0] == 0  # car
    assert cat_indices[1] == 3  # person


def test_read_annotation_skips_no_box2d():
    """read_annotation skips objects that have no box2d key (e.g. poly2d)."""
    _, boxes, cat_indices = _make_patched_dataset(_FAKE_ANNOTATION_NO_BOX2D)

    # Only the car with box2d should be returned, lane with poly2d is skipped
    assert len(boxes) == 1
    assert boxes[0] == [10, 20, 100, 80]
    assert cat_indices[0] == 0  # car


def test_read_annotation_skips_unknown_category(caplog):
    """read_annotation skips unknown categories and logs a warning.

    :param caplog: pytest log capture fixture
    :type caplog: pytest.LogCaptureFixture
    """

    def _fake_glob(pattern):
        return []

    with patch(
        "perceptionmetrics.datasets.bdd100k.glob", side_effect=_fake_glob
    ), patch("os.path.isdir", return_value=True):
        ds = BDD100KDetectionDataset("/images", "/labels")

    with patch(
        "builtins.open",
        mock_open(read_data=json.dumps(_FAKE_ANNOTATION_UNKNOWN_CATEGORY)),
    ), patch("json.load", return_value=_FAKE_ANNOTATION_UNKNOWN_CATEGORY):
        with caplog.at_level(logging.WARNING, logger="root"):
            boxes, cat_indices = ds.read_annotation("/labels/train/img3.json")

    # Only the car should be returned; helicopter is unknown
    assert len(boxes) == 1
    assert boxes[0] == [5, 10, 50, 40]
    assert cat_indices[0] == 0  # car

    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("helicopter" in msg for msg in warning_messages)
