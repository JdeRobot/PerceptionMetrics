from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

# Canonical metrics module — same import used by torch_segmentation.py
import perceptionmetrics.utils.segmentation_metrics as um


def load_mask(file) -> np.ndarray:
    """Load a segmentation mask PNG and return a 2-D integer array.

    Each pixel value is interpreted as a class index (label), matching
    the convention used by the dataset loaders in this repository
    (grayscale PNG where pixel intensity == class ID).

    :param file: File-like object (e.g. Streamlit UploadedFile) or path.
    :return: 2-D numpy array of dtype int32.
    :rtype: np.ndarray
    """
    img = Image.open(file).convert("L")  # grayscale → one channel
    return np.array(img, dtype=np.int32)


def compute_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Create an RGB error-map image from GT and prediction masks.

    - Green (0, 200, 0)  → correctly predicted pixels
    - Red   (200, 0, 0)  → incorrectly predicted pixels

    :param gt:   Ground-truth mask, shape (H, W), integer dtype.
    :param pred: Prediction mask,   shape (H, W), integer dtype.
    :return: HxWx3 uint8 array suitable for ``st.image()``.
    :rtype: np.ndarray
    """
    H, W = gt.shape
    error_map = np.zeros((H, W, 3), dtype=np.uint8)
    error_map[gt == pred] = [0, 200, 0]  # correct → green
    error_map[gt != pred] = [200, 0, 0]  # wrong   → red
    return error_map


def build_ontology_from_masks(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, dict]:
    """Derive a minimal ontology dict from the unique class IDs in GT.

    The resulting dict follows the repository convention expected by
    ``SegmentationMetricsFactory`` and ``get_metrics_dataframe()``:

    .. code-block:: python

        {
            "class_0": {"idx": 0},
            "class_1": {"idx": 1},
            ...
        }

    Class IDs present only in the prediction (but not in GT) are also
    included so the confusion matrix has the correct dimensions.

    :param gt:   Ground-truth mask array.
    :param pred: Prediction mask array.
    :return: Ontology dictionary.
    :rtype: dict
    """
    all_ids = np.union1d(np.unique(gt), np.unique(pred))
    # n_classes must cover the maximum index so the confusion matrix is wide enough
    return {f"class_{int(cid)}": {"idx": int(cid)} for cid in sorted(all_ids)}


def compute_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    ontology: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """Compute segmentation metrics using the repository's canonical factory.

    This is the **only** place where metrics are calculated.  It replicates
    the exact workflow used in ``TorchImageSegmentationModel.eval()``:

    1. Instantiate ``SegmentationMetricsFactory(n_classes)``
    2. Call ``factory.update(pred, gt)`` with integer numpy arrays
    3. Return ``get_metrics_dataframe(factory, ontology)``

    No custom confusion-matrix or accuracy arithmetic is used.

    :param gt:       Ground-truth mask, shape (H, W), integer dtype.
    :param pred:     Prediction mask,   shape (H, W), integer dtype.
    :param ontology: Optional ontology dict.  If ``None``, one is built
                     automatically via ``build_ontology_from_masks()``.
    :return: DataFrame matching the output of ``get_metrics_dataframe()``,
             indexed by class name with columns for every metric in
             ``SegmentationMetricsFactory.METRIC_NAMES``.
    :rtype: pd.DataFrame
    """
    if ontology is None:
        ontology = build_ontology_from_masks(gt, pred)

    # n_classes = highest class index + 1, consistent with how the models
    # initialise the factory (self.n_classes comes from the ontology length).
    n_classes = max(d["idx"] for d in ontology.values()) + 1

    # Step 1 — initialise (mirrors: metrics_factory = um.SegmentationMetricsFactory(self.n_classes))
    factory = um.SegmentationMetricsFactory(n_classes)

    # Step 2 — update (mirrors: metrics_factory.update(pred, label, valid_mask))
    #   SegmentationMetricsFactory.update() asserts integer dtype.
    factory.update(pred.astype(np.int32), gt.astype(np.int32))

    # Step 3 — build DataFrame (mirrors: return um.get_metrics_dataframe(metrics_factory, self.ontology))
    return um.get_metrics_dataframe(factory, ontology)
