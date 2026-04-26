from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from perceptionmetrics.utils.cache import (
    CacheWriter,
    CacheReader,
    CacheStaleError,
    is_cache_valid,
)

HASH_A = "deadbeef1234567890abcdef" * 2
HASH_B = "cafebabe0987654321fedcba" * 2


def _fake_data(n: int = 5, C: int = 3, H: int = 64, W: int = 64) -> list:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        n_det = int(rng.integers(1, 8))
        rows.append(
            (
                f"img_{i:06d}",
                rng.random((C, H, W)).astype("float32"),
                rng.random((n_det, 4)).astype("float32"),
                rng.integers(0, 80, size=(n_det,)).astype("int64"),
                rng.random((n_det,)).astype("float32"),
            )
        )
    return rows


def test_round_trip() -> None:
    data = _fake_data(5)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.hdf5"
        with CacheWriter(path, "yolov8n", "val2017", HASH_A) as w:
            for row in data:
                w.write_image(*row)

        with CacheReader(path, HASH_A) as r:
            assert len(r) == 5
            for img_id, t, bb, lb, sc in data:
                np.testing.assert_array_equal(r.read_tensor(img_id), t)
                p = r.read_preds(img_id)
                np.testing.assert_array_equal(p["bboxes"], bb)
                np.testing.assert_array_equal(p["labels"], lb)
                np.testing.assert_array_almost_equal(p["scores"], sc)
    print("  test_round_trip -----PASSED-----")


def test_stale_raises() -> None:

    data = _fake_data(2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.hdf5"
        with CacheWriter(path, "yolov8n", "val2017", HASH_A) as w:
            for row in data:
                w.write_image(*row)

        with pytest.raises(CacheStaleError, match="model_hash"):
            with CacheReader(path, HASH_B) as r:
                pass
    print("  test_stale_raises -----PASSED-----")


def test_is_cache_valid() -> None:
    data = _fake_data(2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.hdf5"
        with CacheWriter(path, "yolov8n", "val2017", HASH_A) as w:
            for row in data:
                w.write_image(*row)

        assert is_cache_valid(path, HASH_A) is True
        assert is_cache_valid(path, HASH_B) is False
        assert is_cache_valid(Path(tmp) / "missing.hdf5", HASH_A) is False
    print("  test_cache_valid -----PASSED-----")


def test_zero_detection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.hdf5"
        with CacheWriter(path, "yolov8n", "val2017", HASH_A) as w:
            w.write_image(
                "zeros",
                np.zeros((3, 64, 64), dtype="float32"),
                np.zeros((0, 4), dtype="float32"),
                np.zeros((0,), dtype="int64"),
                np.zeros((0,), dtype="float32"),
            )
        with CacheReader(path, HASH_A) as r:
            p = r.read_preds("zeros")
        assert p["bboxes"].shape == (0, 4)
        assert p["labels"].shape == (0,)
        assert p["scores"].shape == (0,)
    print("  test_zero_detection -----PASSED-----")


def test_metadata_round_trip() -> None:
    data = _fake_data(3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.hdf5"
        with CacheWriter(path, "yolov8s", "train2017", HASH_A) as w:
            for row in data:
                w.write_image(*row)
        with CacheReader(path, HASH_A) as r:
            meta = r.metadata()
        assert meta["model_name"] == "yolov8s"
        assert meta["coco_split"] == "train2017"
        assert int(meta["num_images"]) == 3
        assert "timestamp" in meta

    print("  test_metadata_round_trip -----PASSED-----")


def test_image_ids() -> None:
    data = _fake_data(4)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.hdf5"
        with CacheWriter(path, "yolov8n", "val2017", HASH_A) as w:
            for row in data:
                w.write_image(*row)
        with CacheReader(path, HASH_A) as r:
            stored = set(r.image_ids())

    expected = {row[0] for row in data}
    assert stored == expected
    print("  test_images_ids -----PASSED-----")


def main() -> None:
    sep = "=" * 54
    print(f"\n{sep}")
    print(" JdeRobot PerceptionMetrics — HDF5 Cache Tests")
    print(sep + "\n")
    test_round_trip()
    test_stale_raises()
    test_is_cache_valid()
    test_zero_detection()
    test_metadata_round_trip()
    test_image_ids()
    print(f"\n{sep}")
    print(" ALL TESTS PASSED — cache.py Layer 1 verified.")
    print(sep)


if __name__ == "__main__":
    main()
