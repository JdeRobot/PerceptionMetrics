from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch

_SCHEMA_VERSION: str = "1.0"


def compute_model_hash(
    model: torch.nn.Module | None = None,
    file_path: str | None = None,
) -> str:
    """Compute a stable string identifying the model's weights."""
    if file_path is not None:
        sha = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65_536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    if model is not None:
        total = sum(p.numel() for p in model.parameters())
        return f"numel_proxy_{total}"

    raise ValueError("Provide either 'file_path' or 'model' to compute a hash.")


def is_cache_valid(
    path: Union[str, Path],
    expected_model_hash: str,
) -> bool:
    """Quick guard: return True iff cache exists and hashes match."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            meta = f["metadata"].attrs
            return (
                str(meta.get("model_hash", "")) == expected_model_hash
                and str(meta.get("schema_version", "")) == _SCHEMA_VERSION
            )
    except Exception:
        return False


class CacheWriter:
    """Write baseline inference outputs to HDF5 cache file."""

    def __init__(
        self,
        path: Union[str, Path],
        model_name: str,
        coco_split: str,
        model_hash: str,
    ) -> None:
        self.path = Path(path)
        self.model_name = model_name
        self.coco_split = coco_split
        self.model_hash = model_hash
        self._file: h5py.File | None = None
        self._n_written: int = 0

    def __enter__(self) -> "CacheWriter":
        """Open HDF5 file and create metadata groups."""
        self._file = h5py.File(self.path, "w")
        self._write_metadata()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Close file, flush all writes, update image count."""
        if self._file is not None:
            self._file["metadata"].attrs["num_images"] = self._n_written
            self._file.close()
            self._file = None
        return False

    def _write_metadata(self) -> None:
        """Create /metadata group with attributes and empty /tensors, /preds groups."""
        grp = self._file.create_group("metadata")
        grp.attrs["model_name"] = self.model_name
        grp.attrs["coco_split"] = self.coco_split
        grp.attrs["num_images"] = 0
        grp.attrs["timestamp"] = datetime.now(timezone.utc).isoformat()
        grp.attrs["model_hash"] = self.model_hash
        grp.attrs["schema_version"] = _SCHEMA_VERSION

        self._file.create_group("tensors")
        self._file.create_group("preds")

    def write_image(
        self,
        img_id: Union[int, str],
        tensor: Union[torch.Tensor, np.ndarray],
        bboxes: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        scores: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Write one image's tensor and detections to cache."""
        if self._file is None:
            raise RuntimeError(
                "write_image() must be called inside a 'with CacheWriter' block."
            )

        key = str(img_id)

        # Image tensor
        t_np = _to_numpy(tensor, dtype="float32")
        C, H, W = t_np.shape
        self._file["tensors"].create_dataset(
            key,
            data=t_np,
            dtype="float32",
            chunks=(C, H, W),
        )

        # Detection predictions
        pred_grp = self._file["preds"].create_group(key)
        bboxes_np = _to_numpy(bboxes, dtype="float32")
        labels_np = _to_numpy(labels, dtype="int64")
        scores_np = _to_numpy(scores, dtype="float32")

        N = bboxes_np.shape[0]
        pred_grp.create_dataset("bboxes", data=bboxes_np, shape=(N, 4), dtype="float32")
        pred_grp.create_dataset("labels", data=labels_np, shape=(N,), dtype="int64")
        pred_grp.create_dataset("scores", data=scores_np, shape=(N,), dtype="float32")

        self._n_written += 1

    def __repr__(self) -> str:
        return (
            f"CacheWriter(path={self.path}, model={self.model_name!r}, "
            f"split={self.coco_split!r}, written={self._n_written})"
        )


class CacheReader:
    """Read and validate HDF5 cache file."""

    def __init__(
        self,
        path: Union[str, Path],
        expected_model_hash: str,
    ) -> None:
        self.path = Path(path)
        self.expected_model_hash = expected_model_hash
        self._file: h5py.File | None = None

    def __enter__(self) -> "CacheReader":
        """Open cache file and validate schema/hash."""
        if not self.path.exists():
            raise FileNotFoundError(f"Cache file not found: {self.path}")
        self._file = h5py.File(self.path, "r")
        self._validate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Close file."""
        if self._file is not None:
            self._file.close()
            self._file = None
        return False

    def _validate(self) -> None:
        """Raise CacheStaleError on hash or schema mismatch."""
        meta = self._file["metadata"].attrs
        stored_hash = str(meta.get("model_hash", ""))
        if stored_hash != self.expected_model_hash:
            raise CacheStaleError(
                f"model_hash mismatch: stored={stored_hash!r}, "
                f"expected={self.expected_model_hash!r}.  Rebuild the cache."
            )

        stored_ver = str(meta.get("schema_version", ""))
        if stored_ver != _SCHEMA_VERSION:
            raise CacheStaleError(
                f"schema_version mismatch: cache={stored_ver!r}, "
                f"current={_SCHEMA_VERSION!r}.  Rebuild the cache."
            )

    def read_tensor(self, img_id: Union[int, str]) -> np.ndarray:
        """Read preprocessed image tensor for img_id."""
        return self._file[f"tensors/{img_id}"][:]

    def read_preds(self, img_id: Union[int, str]) -> dict:
        """Read detection predictions for img_id."""
        grp = self._file[f"preds/{img_id}"]
        return {
            "bboxes": grp["bboxes"][:],
            "labels": grp["labels"][:],
            "scores": grp["scores"][:],
        }

    def image_ids(self) -> list[str]:
        """All image IDs in the cache."""
        return list(self._file["tensors"].keys())

    def metadata(self) -> dict:
        """Return /metadata attributes as dict."""
        return dict(self._file["metadata"].attrs)

    def __len__(self) -> int:
        """Number of images in cache."""
        return int(self._file["metadata"].attrs["num_images"])

    def __repr__(self) -> str:
        return f"CacheReader(path={self.path}, n={len(self)})"


def _to_numpy(
    x: Union[torch.Tensor, np.ndarray],
    dtype: str,
) -> np.ndarray:
    """Coerce tensor or ndarray to numpy with specified dtype."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)
    return np.asarray(x, dtype=dtype)


class CacheStaleError(Exception):
    """Raised when cache model_hash or schema_version does not match.

    Signals that the cache was built with a different model or older
    schema and must be rebuilt by re-running baseline inference.
    """
