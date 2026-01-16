from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


class DetectionMetricsFactory:
    """Factory class for computing detection metrics including precision, recall, AP, and mAP.

    :param iou_threshold: IoU threshold for matching predictions to ground truth, defaults to 0.5
    :type iou_threshold: float, optional
    :param num_classes: Number of classes in the dataset, defaults to None
    :type num_classes: Optional[int], optional
    """

    def __init__(self, iou_threshold: float = 0.5, num_classes: Optional[int] = None):
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.results = defaultdict(list)  # stores detection results per class
        # Store raw data for multi-threshold evaluation
        self.raw_data = (
            []
        )  # List of (gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)
        self.gt_counts = defaultdict(int)  # Count of GT instances per class

    def update(self, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores):
        """Add a batch of predictions and ground truths.

        :param gt_boxes: Ground truth bounding boxes, shape (num_gt, 4)
        :type gt_boxes: List[ndarray]
        :param gt_labels: Ground truth class labels
        :type gt_labels: List[int]
        :param pred_boxes: Predicted bounding boxes, shape (num_pred, 4)
        :type pred_boxes: List[ndarray]
        :param pred_labels: Predicted class labels
        :type pred_labels: List[int]
        :param pred_scores: Prediction confidence scores
        :type pred_scores: List[float]
        """

        # Convert torch tensors to numpy
        if hasattr(gt_boxes, "detach"):
            gt_boxes = gt_boxes.detach().cpu().numpy()
        if hasattr(gt_labels, "detach"):
            gt_labels = gt_labels.detach().cpu().numpy()
        if hasattr(pred_boxes, "detach"):
            pred_boxes = pred_boxes.detach().cpu().numpy()
        if hasattr(pred_labels, "detach"):
            pred_labels = pred_labels.detach().cpu().numpy()
        if hasattr(pred_scores, "detach"):
            pred_scores = pred_scores.detach().cpu().numpy()

        # Store raw data for multi-threshold evaluation
        self.raw_data.append(
            (gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)
        )

        # Handle empty inputs
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return  # Nothing to process

        # Handle case where there are predictions but no ground truth
        if len(gt_boxes) == 0:
            for p_label, score in zip(pred_labels, pred_scores):
                self.results[p_label].append((score, 0))  # All are false positives
            return

        # Handle case where there is ground truth but no predictions
        if len(pred_boxes) == 0:
            for g_label in gt_labels:
                self.results[g_label].append((None, -1))  # All are false negatives
            return

        matches = self._match_predictions(
            gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores
        )

        for label in matches:
            self.results[label].extend(matches[label])

        # Update ground truth counts
        for g_label in gt_labels:
            self.gt_counts[int(g_label)] += 1

    def _match_predictions(
        self,
        gt_boxes: np.ndarray,
        gt_labels: List[int],
        pred_boxes: np.ndarray,
        pred_labels: List[int],
        pred_scores: List[float],
        iou_threshold: Optional[float] = None,
    ) -> Dict[int, List[Tuple[float, int]]]:
        """Match predictions to ground truth and return per-class TP/FP flags with scores.

        :param gt_boxes: Ground truth bounding boxes, shape (num_gt, 4)
        :type gt_boxes: np.ndarray
        :param gt_labels: Ground truth class labels
        :type gt_labels: List[int]
        :param pred_boxes: Predicted bounding boxes, shape (num_pred, 4)
        :type pred_boxes: np.ndarray
        :param pred_labels: Predicted class labels
        :type pred_labels: List[int]
        :param pred_scores: Prediction confidence scores
        :type pred_scores: List[float]
        :param iou_threshold: IoU threshold for matching, overrides self.iou_threshold if provided, defaults to None
        :type iou_threshold: Optional[float], optional
        :return: Dictionary mapping class labels to list of (score, tp_or_fp) tuples
        :rtype: Dict[int, List[Tuple[float, int]]]
        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        results = defaultdict(list)
        used = set()

        ious = compute_iou_matrix(pred_boxes, gt_boxes)  # shape: (num_preds, num_gts)

        for i, (p_box, p_label, score) in enumerate(
            zip(pred_boxes, pred_labels, pred_scores)
        ):
            max_iou = 0
            max_j = -1

            for j, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
                if j in used or p_label != g_label:
                    continue
                iou = ious[i, j]
                if iou > max_iou:
                    max_iou = iou
                    max_j = j

            if max_iou >= iou_threshold:
                results[p_label].append((score, 1))  # True positive
                used.add(max_j)
            else:
                results[p_label].append((score, 0))  # False positive

        # Handle false negatives (missed GTs)
        for j, g_label in enumerate(gt_labels):
            if j not in used:
                results[g_label].append((None, -1))  # FN, no score

        return results

    def compute_metrics(self) -> Dict[int, Dict[str, float]]:
        """Compute per-class precision, recall, AP, and mAP.

        :return: Dictionary mapping class IDs to metric dictionaries, plus mAP under key -1
        :rtype: Dict[int, Dict[str, float]]
        """
        metrics = {}
        ap_values = []

        for label, detections in self.results.items():
            # Skip classes with no ground truth instances
            if self.gt_counts.get(int(label), 0) == 0:
                continue

            detections = sorted(
                [d for d in detections if d[0] is not None], key=lambda x: -x[0]
            )
            scores = [d[0] for d in detections]
            tps = [d[1] == 1 for d in detections]
            fps = [d[1] == 0 for d in detections]
            fn_count = sum(1 for d in self.results[label] if d[1] == -1)

            ap, precision, recall = compute_ap(tps, fps, fn_count)

            metrics[label] = {
                "AP": ap,
                "Precision": precision[-1] if len(precision) > 0 else 0,
                "Recall": recall[-1] if len(recall) > 0 else 0,
                "TP": sum(tps),
                "FP": sum(fps),
                "FN": fn_count,
            }

            ap_values.append(ap)

        # Add mAP (mean over all class APs)
        if ap_values:
            metrics[-1] = {
                "AP": np.mean(ap_values),
                "Precision": np.nan,
                "Recall": np.nan,
                "TP": np.nan,
                "FP": np.nan,
                "FN": np.nan,
            }

        return metrics

    def compute_coco_map(self) -> float:
        """Compute COCO-style mAP (mean AP over IoU thresholds 0.5:0.05:0.95).

        :return: mAP@[0.5:0.95]
        :rtype: float
        """
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        aps = []

        for iou_thresh in iou_thresholds:
            # Reset results for this threshold
            threshold_results = defaultdict(list)

            # Process all raw data with current threshold
            for (
                gt_boxes,
                gt_labels,
                pred_boxes,
                pred_labels,
                pred_scores,
            ) in self.raw_data:
                # Handle empty inputs
                if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                    continue

                # Handle case where there are predictions but no ground truth
                if len(gt_boxes) == 0:
                    for p_label, score in zip(pred_labels, pred_scores):
                        threshold_results[p_label].append(
                            (score, 0)
                        )  # All are false positives
                    continue

                # Handle case where there is ground truth but no predictions
                if len(pred_boxes) == 0:
                    for g_label in gt_labels:
                        threshold_results[g_label].append(
                            (None, -1)
                        )  # All are false negatives
                    continue

                matches = self._match_predictions(
                    gt_boxes,
                    gt_labels,
                    pred_boxes,
                    pred_labels,
                    pred_scores,
                    iou_thresh,
                )

                for label in matches:
                    threshold_results[label].extend(matches[label])

            # Compute AP for this threshold
            threshold_ap_values = []
            for label, detections in threshold_results.items():
                # Skip classes with no ground truth instances
                if self.gt_counts.get(int(label), 0) == 0:
                    continue

                detections = sorted(
                    [d for d in detections if d[0] is not None], key=lambda x: -x[0]
                )
                tps = [d[1] == 1 for d in detections]
                fps = [d[1] == 0 for d in detections]
                fn_count = sum(1 for d in threshold_results[label] if d[1] == -1)

                ap, _, _ = compute_ap(tps, fps, fn_count)
                threshold_ap_values.append(ap)

            # Mean AP for this threshold
            if threshold_ap_values:
                aps.append(np.mean(threshold_ap_values))
            else:
                aps.append(0.0)

        # Return mean over all thresholds
        return np.mean(aps) if aps else 0.0

    def get_overall_precision_recall_curve(self) -> Dict[str, List[float]]:
        """Get overall precision-recall curve data (aggregated across all classes).

        :return: Dictionary with 'precision' and 'recall' keys containing curve data
        :rtype: Dict[str, List[float]]
        """
        all_detections = []

        # Collect all detections from all classes
        for label, detections in self.results.items():
            all_detections.extend(detections)

        if len(all_detections) == 0:
            return {"precision": [0.0], "recall": [0.0]}

        # Sort by score
        all_detections = sorted(
            [d for d in all_detections if d[0] is not None], key=lambda x: -x[0]
        )

        tps = [d[1] == 1 for d in all_detections]
        fps = [d[1] == 0 for d in all_detections]
        fn_count = sum(1 for d in all_detections if d[1] == -1)

        _, precision, recall = compute_ap(tps, fps, fn_count)

        return {
            "precision": (
                precision.tolist() if hasattr(precision, "tolist") else list(precision)
            ),
            "recall": recall.tolist() if hasattr(recall, "tolist") else list(recall),
        }

    def compute_auc_pr(self) -> float:
        """Compute the Area Under the Precision-Recall Curve (AUC-PR).

        :return: Area under the precision-recall curve
        :rtype: float
        """
        curve_data = self.get_overall_precision_recall_curve()
        precision = np.array(curve_data["precision"])
        recall = np.array(curve_data["recall"])

        # Handle edge cases
        if len(precision) == 0 or len(recall) == 0:
            return 0.0

        # Sort by recall to ensure proper integration
        sorted_indices = np.argsort(recall)
        recall_sorted = recall[sorted_indices]
        precision_sorted = precision[sorted_indices]

        # Compute AUC using trapezoidal rule
        auc = np.trapz(precision_sorted, recall_sorted)

        return float(auc)

    def get_metrics_dataframe(self, ontology: dict) -> pd.DataFrame:
        """Get results as a pandas DataFrame.

        :param ontology: Mapping from class name → { "idx": int }
        :type ontology: dict
        :return: DataFrame with metrics as rows and classes as columns (with mean)
        :rtype: pd.DataFrame
        """
        all_metrics = self.compute_metrics()
        # Build a dict: metric -> {class_name: value}
        metrics_dict = {}
        class_names = list(ontology.keys())

        for metric in ["AP", "Precision", "Recall", "TP", "FP", "FN"]:
            metrics_dict[metric] = {}
            for class_name, class_data in ontology.items():
                idx = class_data["idx"]
                value = all_metrics.get(idx, {}).get(metric, np.nan)
                metrics_dict[metric][class_name] = value
            # Compute mean (ignore NaN for mean)
            values = [v for v in metrics_dict[metric].values() if not pd.isna(v)]
            metrics_dict[metric]["mean"] = np.mean(values) if values else np.nan

        # Add COCO-style mAP
        coco_map = self.compute_coco_map()
        metrics_dict["mAP@[0.5:0.95]"] = {}
        for class_name in class_names:
            metrics_dict["mAP@[0.5:0.95]"][
                class_name
            ] = np.nan  # Per-class not applicable
        metrics_dict["mAP@[0.5:0.95]"]["mean"] = coco_map

        # Add AUC-PR
        auc_pr = self.compute_auc_pr()
        metrics_dict["AUC-PR"] = {}
        for class_name in class_names:
            metrics_dict["AUC-PR"][class_name] = np.nan  # Per-class not applicable
        metrics_dict["AUC-PR"]["mean"] = auc_pr

        df = pd.DataFrame(metrics_dict)
        return df.T  # metrics as rows, classes as columns (with mean)


def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between pred and gt boxes.

    :param pred_boxes: Predicted bounding boxes, shape (num_pred, 4)
    :type pred_boxes: np.ndarray
    :param gt_boxes: Ground truth bounding boxes, shape (num_gt, 4)
    :type gt_boxes: np.ndarray
    :return: IoU matrix with shape (num_pred, num_gt)
    :rtype: np.ndarray
    """
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pb, gb)
    return iou_matrix


def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes.

    :param boxA: First bounding box [x1, y1, x2, y2]
    :type boxA: array-like
    :param boxB: Second bounding box [x1, y1, x2, y2]
    :type boxB: array-like
    :return: IoU value between 0 and 1
    :rtype: float
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def compute_ap(tps, fps, fn):
    """Compute Average Precision (AP) using VOC-style 11-point interpolation.

    :param tps: List of true positive flags
    :type tps: List[bool] or np.ndarray
    :param fps: List of false positive flags
    :type fps: List[bool] or np.ndarray
    :param fn: Number of false negatives
    :type fn: int
    :return: Tuple of (AP, precision array, recall array)
    :rtype: Tuple[float, np.ndarray, np.ndarray]
    """
    tps = np.array(tps, dtype=np.float32)
    fps = np.array(fps, dtype=np.float32)

    # Handle edge cases
    if len(tps) == 0:
        if fn == 0:
            return 1.0, [1.0], [1.0]  # Perfect case: no predictions, no ground truth
        else:
            return 0.0, [0.0], [0.0]  # No predictions but there was ground truth

    tp_cumsum = np.cumsum(tps)
    fp_cumsum = np.cumsum(fps)

    if tp_cumsum.size:
        denom = tp_cumsum[-1] + fn
        if denom > 0:
            recalls = tp_cumsum / denom
        else:
            recalls = np.zeros_like(tp_cumsum)
    else:
        recalls = []

    # Compute precision with proper handling of division by zero
    denominator = tp_cumsum + fp_cumsum
    precisions = np.where(denominator > 0, tp_cumsum / denominator, 0.0)

    # VOC-style 11-point interpolation
    ap = 0
    for r in np.linspace(0, 1, 11):
        p = [p for p, rc in zip(precisions, recalls) if rc >= r]
        ap += max(p) if p else 0
    ap /= 11.0

    return ap, precisions, recalls
