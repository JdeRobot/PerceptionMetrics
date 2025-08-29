from collections import defaultdict
import math
from typing import Optional

import numpy as np
import pandas as pd


class SegmentationMetricsFactory:
    """'Factory' class to accumulate results and compute metrics for segmentation tasks

    :param n_classes: Number of classes to evaluate
    :type n_classes: int
    """

    METRIC_NAMES = [
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "accuracy",
        "f1_score",
        "iou",
    ]

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(
        self, pred: np.ndarray, gt: np.ndarray, valid_mask: Optional[np.ndarray] = None
    ):
        """Accumulate results for a new batch

        :param pred: Array containing prediction
        :type pred: np.ndarray
        :param gt: Array containing ground truth
        :type gt: np.ndarray
        :param valid_mask: Binary mask where False elements will be igonred, defaults to None
        :type valid_mask: Optional[np.ndarray], optional
        """
        assert pred.shape == gt.shape, "Prediction and GT shapes don't match"
        assert np.issubdtype(pred.dtype, np.integer), "Prediction should be integer"
        assert np.issubdtype(gt.dtype, np.integer), "GT should be integer"

        # Build mask of valid elements
        mask = (gt >= 0) & (gt < self.n_classes)
        if valid_mask is not None:
            mask &= valid_mask

        # Update confusion matrix
        if np.count_nonzero(gt >= 16):
            pass

        # Update confusion matrix
        new_entry = np.bincount(
            self.n_classes * gt[mask].astype(int) + pred[mask].astype(int),
            minlength=self.n_classes**2,
        )
        self.confusion_matrix += new_entry.reshape(self.n_classes, self.n_classes)

    def get_metric_names(self) -> list[str]:
        """Get available metric names

        :return: List of available metric names
        :rtype: list[str]
        """
        return self.METRIC_NAMES

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix

        :return: Confusion matrix
        :rtype: np.ndarray
        """
        return self.confusion_matrix

    def get_tp(self, per_class: bool = True) -> np.ndarray | int:
        """True Positives

        :param per_class: Return per class TP, defaults to True
        :type per_class: bool, optional
        :return: True Positives
        :rtype: np.ndarray | int
        """
        tp = np.diag(self.confusion_matrix)
        return tp if per_class else int(np.nansum(tp))

    def get_fp(self, per_class: bool = True) -> np.ndarray | int:
        """False Positives

        :param per_class: Return per class FP, defaults to True
        :type per_class: bool, optional
        :return: True Positives
        :rtype: np.ndarray | int
        """
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        return fp if per_class else int(np.nansum(fp))

    def get_fn(self, per_class: bool = True) -> np.ndarray | int:
        """False negatives

        :param per_class: Return per class FN, defaults to True
        :type per_class: bool, optional
        :return: True Positives
        :rtype: np.ndarray | int
        """
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        return fn if per_class else int(np.nansum(fn))

    def get_tn(self, per_class: bool = True) -> np.ndarray | int:
        """True negatives

        :param per_class: Return per class TN, defaults to True
        :type per_class: bool, optional
        :return: True Positives
        :rtype: np.ndarray | int
        """
        total = self.confusion_matrix.sum()
        tn = total - (self.get_tp() + self.get_fp() + self.get_fn())
        return tn if per_class else int(np.nansum(tn))

    def get_precision(self, per_class: bool = True) -> np.ndarray | float:
        """Precision = TP / (TP + FP)

        :param per_class: Return per class precision, defaults to True
        :type per_class: bool, optional
        :return: True Positives
        :rtype: np.ndarray | int
        """
        tp = self.get_tp(per_class)
        fp = self.get_fp(per_class)
        denominator = tp + fp

        if np.isscalar(denominator):
            return float(tp / denominator) if denominator > 0 else math.nan
        else:
            return np.where(denominator > 0, tp / denominator, np.nan)

    def get_recall(self, per_class: bool = True) -> np.ndarray | float:
        """Recall = TP / (TP + FN)

        :param per_class: Return per class recall, defaults to True
        :type per_class: bool, optional
        :return: True Positives
        :rtype: np.ndarray | int
        """
        tp = self.get_tp(per_class)
        fn = self.get_fn(per_class)
        denominator = tp + fn

        if np.isscalar(denominator):
            return float(tp / denominator) if denominator > 0 else math.nan
        else:
            return np.where(denominator > 0, tp / denominator, np.nan)

    def get_accuracy(self, per_class: bool = True) -> np.ndarray | float:
        """Accuracy = (TP + TN) / (TP + FP + FN + TN)

        :param per_class: Return per class accuracy, defaults to True
        :type per_class: bool, optional
        :return: True Positives
        :rtype: np.ndarray | int
        """
        tp = self.get_tp(per_class)
        fp = self.get_fp(per_class)
        fn = self.get_fn(per_class)
        tn = self.get_tn(per_class)
        total = tp + fp + fn + tn

        if np.isscalar(total):
            return float((tp + tn) / total) if total > 0 else math.nan
        else:
            return np.where(total > 0, (tp + tn) / total, np.nan)

    def get_f1_score(self, per_class: bool = True) -> np.ndarray | float:
        """F1-score = 2 * (Precision * Recall) / (Precision + Recall)

        :param per_class: Return per class F1 score, defaults to True
        :type per_class: bool, optional
        :return: True Positives
        :rtype: np.ndarray | int
        """
        precision = self.get_precision(per_class)
        recall = self.get_recall(per_class)
        denominator = precision + recall

        if np.isscalar(denominator):
            return (
                2 * (precision * recall) / denominator if denominator > 0 else math.nan
            )
        else:
            return np.where(
                denominator > 0, 2 * (precision * recall) / denominator, np.nan
            )

    def get_iou(self, per_class: bool = True) -> np.ndarray | float:
        """IoU = TP / (TP + FP + FN)

        :param per_class: Return per class IoU, defaults to True
        :type per_class: bool, optional
        :return: True Positives
        :rtype: np.ndarray | int
        """
        tp = self.get_tp(per_class)
        fp = self.get_fp(per_class)
        fn = self.get_fn(per_class)
        union = tp + fp + fn

        if np.isscalar(union):
            return float(tp / union) if union > 0 else math.nan
        else:
            return np.where(union > 0, tp / union, np.nan)

    def get_averaged_metric(
        self, metric_name: str, method: str, weights: Optional[np.ndarray] = None
    ) -> float:
        """Get average metric value

        :param metric: Name of the metric to compute
        :type metric: str
        :param method: Method to use for averaging ('macro', 'micro' or 'weighted')
        :type method: str
        :param weights: Weights for weighted averaging, defaults to None
        :type weights: Optional[np.ndarray], optional
        :return: Average metric value
        :rtype: float
        """
        metric = getattr(self, f"get_{metric_name}")
        if method == "macro":
            return float(np.nanmean(metric(per_class=True)))
        if method == "micro":
            return float(metric(per_class=False))
        if method == "weighted":
            assert (
                weights is not None
            ), "Weights should be provided for weighted averaging"
            return float(np.nansum(metric(per_class=True) * weights))
        raise ValueError(f"Unknown method {method}")

    def get_metric_per_name(
        self, metric_name: str, per_class: bool = True
    ) -> np.ndarray | float | int:
        """Get metric value by name

        :param metric_name: Name of the metric to compute
        :type metric_name: str
        :param per_class: Return per class metric, defaults to True
        :type per_class: bool, optional
        :return: Metric value
        :rtype: np.ndarray | float | int
        """
        return getattr(self, f"get_{metric_name}")(per_class=per_class)


def get_metrics_dataframe(
    metrics_factory: SegmentationMetricsFactory, ontology: dict
) -> pd.DataFrame:
    """Build a DataFrame with all metrics (global and per class) plus confusion matrix

    :param metrics_factory: Properly updated SegmentationMetricsFactory object
    :type metrics_factory: SegmentationMetricsFactory
    :param ontology: Ontology dictionary
    :type ontology: dict
    :return: DataFrame with all metrics
    :rtype: pd.DataFrame
    """
    # Build results dataframe
    results = defaultdict(dict)

    # Add per class and global metrics
    for metric in metrics_factory.get_metric_names():
        per_class = metrics_factory.get_metric_per_name(metric, per_class=True)

        for class_name, class_data in ontology.items():
            results[class_name][metric] = float(per_class[class_data["idx"]])

        if metric not in ["tp", "fp", "fn", "tn"]:
            for avg_method in ["macro", "micro"]:
                results[avg_method][metric] = metrics_factory.get_averaged_metric(
                    metric, avg_method
                )

    # Add confusion matrix
    for class_name_a, class_data_a in ontology.items():
        for class_name_b, class_data_b in ontology.items():
            results[class_name_a][class_name_b] = metrics_factory.confusion_matrix[
                class_data_a["idx"], class_data_b["idx"]
            ]

    return pd.DataFrame(results)
