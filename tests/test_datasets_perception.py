import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Stub heavy optional C-extensions that are not installed in the test environment
for _stub in ("open3d", "supervision"):
    if _stub not in sys.modules:
        sys.modules[_stub] = MagicMock()

from perceptionmetrics.datasets.perception import PerceptionDataset  # noqa: E402


class _DummyDataset(PerceptionDataset):
    """Minimal concrete subclass used to test PerceptionDataset behaviour."""

    def make_fname_global(self):
        pass


def test_validate_splits_raises_for_missing_split():
    """_validate_splits raises ValueError when the requested split is absent.

    Only a 'train' split is loaded; requesting 'test' must raise a descriptive
    ValueError rather than silently returning NaN/0.0 metrics.

    :raises ValueError: Expected when the requested split is not in the dataset.
    """
    dataset = _DummyDataset(
        dataset=pd.DataFrame(
            [{"image": "img1.jpg", "annotation": "lbl1.txt", "split": "train"}]
        ),
        dataset_dir="/fake",
        ontology={},
    )

    # Present split must not raise
    dataset._validate_splits(["train"])

    # Absent split must raise a descriptive ValueError
    with pytest.raises(ValueError, match="test"):
        dataset._validate_splits(["test"])
