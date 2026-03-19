from typing import Any, Iterable, Dict
import numpy as np

from perceptionmetrics.utils.segmentation_metrics import SegmentationMetricsFactory


class SegmentationEvaluator:
    """
    Minimal reusable evaluator for segmentation tasks.
    """

    def __init__(self, n_classes: int):
        self.metrics = SegmentationMetricsFactory(n_classes)

    def reset(self):
        self.metrics.reset()

    def evaluate(self, model: Any, dataset: Iterable[Dict]):
        self.reset()

        for sample in dataset:
            image = sample["image"]
            gt = sample["mask"]

            pred = model.predict(image)

            # Convert to numpy if needed
            if hasattr(pred, "detach"):
                pred = pred.detach().cpu().numpy()
            if hasattr(gt, "detach"):
                gt = gt.detach().cpu().numpy()

            pred = np.asarray(pred)
            gt = np.asarray(gt)

            self.metrics.update(pred, gt)

        return self.metrics