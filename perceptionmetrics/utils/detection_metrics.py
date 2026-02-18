from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


class DetectionMetricsFactory:
    """Factory class for computing detection metrics including precision, recall, AP, and mAP.

    Evaluation Strategy (aligned with Ultralytics):
    - mAP and PR curves: computed without confidence threshold filtering (all predictions kept)
    - Precision, Recall, Confusion Matrix: computed at a specific confidence threshold
      (either user-defined or automatically selected to maximize F1 score)
    - Confusion Matrix includes implicit background class for unmatched predictions/GT

    :param iou_threshold: IoU threshold for matching predictions to ground truth, defaults to 0.5
    :type iou_threshold: float, optional
    :param num_classes: Number of classes in the dataset, defaults to None
    :type num_classes: Optional[int], optional
    :param conf_threshold: Confidence threshold for precision/recall/confusion matrix.
        Set to None to auto-select threshold that maximizes F1 score, defaults to None
    :type conf_threshold: Optional[float], optional
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        num_classes: Optional[int] = None,
        conf_threshold: Optional[float] = None,
    ):
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.optimal_conf_threshold = None  # Will be computed if conf_threshold is None
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

    def _find_optimal_confidence_threshold(self) -> Tuple[float, float]:
        """Find the confidence threshold that maximizes F1 score.

        :return: Tuple of (optimal_threshold, max_f1_score)
        :rtype: Tuple[float, float]
        """
        all_detections = []

        # Collect all detections from all classes with scores
        for label, detections in self.results.items():
            for score, tp_fp_fn in detections:
                if score is not None:  # Exclude FN entries (they have no score)
                    all_detections.append((score, tp_fp_fn, label))

        if len(all_detections) == 0:
            return 0.0, 0.0

        # Sort by score descending
        all_detections.sort(key=lambda x: -x[0])

        # Get all unique scores as candidate thresholds
        unique_scores = sorted(set(d[0] for d in all_detections), reverse=True)

        best_threshold = 0.0
        best_f1 = 0.0

        # Total FN count (all FN entries across all classes)
        total_fn = sum(
            1 for detections in self.results.values() for d in detections if d[1] == -1
        )

        for threshold in unique_scores:
            tp_count = sum(1 for d in all_detections if d[0] >= threshold and d[1] == 1)
            fp_count = sum(1 for d in all_detections if d[0] >= threshold and d[1] == 0)
            fn_count = total_fn + sum(
                1 for d in all_detections if d[0] < threshold and d[1] == 1
            )

            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        return best_threshold, best_f1

    def get_confusion_matrix(
        self, conf_threshold: Optional[float] = None, include_background: bool = True
    ) -> np.ndarray:
        """Compute confusion matrix at specified confidence threshold.

        Following Ultralytics convention: includes implicit background class for unmatched boxes.
        - Unmatched predictions → counted as predicted_class vs background
        - Unmatched ground truth → counted as background vs true_class

        :param conf_threshold: Confidence threshold. If None, uses self.conf_threshold or finds optimal.
        :type conf_threshold: Optional[float]
        :param include_background: Whether to include background class (index -1), defaults to True
        :type include_background: bool
        :return: Confusion matrix of shape (n_classes + 1, n_classes + 1) if include_background else (n_classes, n_classes)
        :rtype: np.ndarray
        """
        if conf_threshold is None:
            if self.conf_threshold is not None:
                conf_threshold = self.conf_threshold
            else:
                # Auto-select optimal threshold
                if self.optimal_conf_threshold is None:
                    self.optimal_conf_threshold, _ = (
                        self._find_optimal_confidence_threshold()
                    )
                conf_threshold = self.optimal_conf_threshold

        if self.num_classes is None:
            # Infer num_classes from data
            all_labels = set()
            for _, gt_labels, _, pred_labels, _ in self.raw_data:
                all_labels.update(gt_labels)
                all_labels.update(pred_labels)
            self.num_classes = max(all_labels) + 1 if all_labels else 0

        # Initialize confusion matrix
        # Background class will be at index num_classes
        matrix_size = self.num_classes + 1 if include_background else self.num_classes
        confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=np.int64)

        # Process raw data
        for gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores in self.raw_data:
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue

            # Filter predictions by confidence threshold
            if len(pred_boxes) > 0:
                conf_mask = pred_scores >= conf_threshold
                pred_boxes_filtered = pred_boxes[conf_mask]
                pred_labels_filtered = pred_labels[conf_mask]
                pred_scores_filtered = pred_scores[conf_mask]
            else:
                pred_boxes_filtered = pred_boxes
                pred_labels_filtered = pred_labels
                pred_scores_filtered = pred_scores

            # Handle case: no GT, but have predictions
            if len(gt_boxes) == 0 and len(pred_boxes_filtered) > 0:
                if include_background:
                    for p_label in pred_labels_filtered:
                        # Predicted as p_label, but actually background
                        confusion_matrix[self.num_classes, int(p_label)] += 1
                continue

            # Handle case: have GT, but no predictions above threshold
            if len(pred_boxes_filtered) == 0 and len(gt_boxes) > 0:
                if include_background:
                    for g_label in gt_labels:
                        # Predicted as background, but actually g_label
                        confusion_matrix[int(g_label), self.num_classes] += 1
                continue

            # Match predictions to ground truth
            if len(pred_boxes_filtered) > 0 and len(gt_boxes) > 0:
                ious = compute_iou_matrix(pred_boxes_filtered, gt_boxes)
                used_gt = set()
                matched_preds = set()

                # Match each prediction to best GT
                for i, (p_label, score) in enumerate(
                    zip(pred_labels_filtered, pred_scores_filtered)
                ):
                    max_iou = 0
                    max_j = -1

                    for j, g_label in enumerate(gt_labels):
                        if j in used_gt or p_label != g_label:
                            continue
                        iou = ious[i, j]
                        if iou > max_iou:
                            max_iou = iou
                            max_j = j

                    if max_iou >= self.iou_threshold:
                        # True positive: correct match
                        confusion_matrix[int(gt_labels[max_j]), int(p_label)] += 1
                        used_gt.add(max_j)
                        matched_preds.add(i)
                    else:
                        # False positive: predicted p_label but no matching GT
                        if include_background:
                            confusion_matrix[self.num_classes, int(p_label)] += 1

                # Handle unmatched GTs (false negatives)
                if include_background:
                    for j, g_label in enumerate(gt_labels):
                        if j not in used_gt:
                            # Missed GT: should be g_label but predicted as background
                            confusion_matrix[int(g_label), self.num_classes] += 1

        return confusion_matrix

    def get_optimal_conf_threshold(self) -> Dict[str, float]:
        """Get the optimal confidence threshold that maximizes F1 score.

        :return: Dictionary with 'threshold' and 'f1_score' keys
        :rtype: Dict[str, float]
        """
        if self.optimal_conf_threshold is None:
            threshold, f1 = self._find_optimal_confidence_threshold()
            self.optimal_conf_threshold = threshold
        else:
            # Recompute F1 at stored threshold
            _, f1 = self._find_optimal_confidence_threshold()

        return {"threshold": self.optimal_conf_threshold, "f1_score": f1}

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
