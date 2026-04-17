from typing import Any, List

import numpy as np
from PIL import Image

import perceptionmetrics.utils.segmentation_metrics as um


class SegmentationEvaluator:
    """Minimal prototype evaluator for segmentation models.

    This class extracts only the common evaluation loop:
    dataset iteration, `model.predict()` invocation, and metric aggregation.

    The current prototype intentionally excludes ontology translation, prediction
    saving, per-sample reports, batching, and backend-specific behavior so it can
    be integrated safely and extended incrementally in the future.

    Note:
        This implementation is intentionally minimal and mirrors only the
        default evaluation path to ensure backward compatibility.
    """

    def __init__(self, n_classes: int):
        """Initialize metric aggregation.

        :param n_classes: Number of segmentation classes.
        :type n_classes: int
        """
        self.n_classes = n_classes
        self.metrics_factory = um.SegmentationMetricsFactory(n_classes)

    def evaluate(self, model: Any, dataset: Any, splits: List[str]):
        """Evaluate a segmentation model over dataset samples.

        :param model: Segmentation model exposing `predict(image)`.
        :type model: Any
        :param dataset: Dataset object exposing `dataset` and `make_fname_global`.
        :type dataset: Any
        :param splits: Dataset splits to evaluate.
        :type splits: List[str]
        :return: Aggregated segmentation metrics factory.
        :rtype: um.SegmentationMetricsFactory
        """
        if "split" not in dataset.dataset.columns:
            raise ValueError("Dataset must contain 'split' column")

        dataset.make_fname_global()
        df = dataset.dataset[dataset.dataset["split"].isin(splits)].copy()

        for _, row in df.iterrows():
            image = Image.open(row["image"]).convert("RGB")
            label = np.array(Image.open(row["label"]))
            pred = model.predict(image)

            pred = np.array(pred)

            self.metrics_factory.update(pred, label, None)

        return self.metrics_factory
