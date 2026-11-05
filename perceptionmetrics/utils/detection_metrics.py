from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from shapely.geometry import Polygon


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

    def reset(self):
        """Reset all accumulated results to initial state.

        This allows reusing the same factory instance for multiple
        independent evaluations without creating a new object.
        """
        self.results = defaultdict(list)
        self.raw_data = []
        self.gt_counts = defaultdict(int)

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

        # Update ground truth counts
        for g_label in gt_labels:
            self.gt_counts[int(g_label)] += 1

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

        # Sort predictions by descending score to ensure highest-confidence predictions match first
        sorted_indices = np.argsort([-s for s in pred_scores])

        for i in sorted_indices:
            p_box = pred_boxes[i]
            p_label = pred_labels[i]
            score = pred_scores[i]
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

        fn_count = sum(1 for d in all_detections if d[1] == -1)
        
        # Sort by score
        all_detections = sorted(
            [d for d in all_detections if d[0] is not None], key=lambda x: -x[0]
        )

        tps = [d[1] == 1 for d in all_detections]
        fps = [d[1] == 0 for d in all_detections]

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
def _bev_corners(box: np.ndarray) -> np.ndarray:
    """Return counter-clockwise XY corners for an oriented LiDAR box."""
    x, y, _, dx, dy, _, yaw = box[:7]
    corners = np.array(
        [
            [-dx / 2, -dy / 2],
            [dx / 2, -dy / 2],
            [dx / 2, dy / 2],
            [-dx / 2, dy / 2],
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float64,
    )
    return corners @ rotation.T + np.array([x, y], dtype=np.float64)


def _bev_intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Return the overlap area between two oriented LiDAR boxes in BEV."""
    polygon_a = Polygon(_bev_corners(box_a))
    polygon_b = Polygon(_bev_corners(box_b))
    return float(polygon_a.intersection(polygon_b).area)


def compute_bev_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute oriented top-down IoU for two LiDAR boxes."""
    box_a = np.asarray(box_a, dtype=np.float64)
    box_b = np.asarray(box_b, dtype=np.float64)
    if box_a.shape[0] < 7 or box_b.shape[0] < 7:
        raise ValueError("LiDAR boxes must contain at least seven values.")
    if np.any(box_a[3:5] <= 0) or np.any(box_b[3:5] <= 0):
        return 0.0

    intersection = _bev_intersection_area(box_a, box_b)
    area_a = box_a[3] * box_a[4]
    area_b = box_b[3] * box_b[4]
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def compute_3d_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute oriented 3D IoU for bottom-center LiDAR boxes."""
    box_a = np.asarray(box_a, dtype=np.float64)
    box_b = np.asarray(box_b, dtype=np.float64)
    if box_a.shape[0] < 7 or box_b.shape[0] < 7:
        raise ValueError("LiDAR boxes must contain at least seven values.")
    if np.any(box_a[3:6] <= 0) or np.any(box_b[3:6] <= 0):
        return 0.0

    intersection_area = _bev_intersection_area(box_a, box_b)
    a_bottom, a_top = box_a[2], box_a[2] + box_a[5]
    b_bottom, b_top = box_b[2], box_b[2] + box_b[5]
    intersection_height = max(0.0, min(a_top, b_top) - max(a_bottom, b_bottom))
    intersection_volume = intersection_area * intersection_height

    volume_a = float(np.prod(box_a[3:6]))
    volume_b = float(np.prod(box_b[3:6]))
    union = volume_a + volume_b - intersection_volume
    return intersection_volume / union if union > 0 else 0.0


def compute_bev_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute oriented BEV IoU for every predicted/ground-truth box pair."""
    return _compute_lidar_iou_matrix(pred_boxes, gt_boxes, compute_bev_iou)


def compute_3d_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute oriented 3D IoU for every predicted/ground-truth box pair."""
    return _compute_lidar_iou_matrix(pred_boxes, gt_boxes, compute_3d_iou)


def _compute_lidar_iou_matrix(pred_boxes, gt_boxes, iou_function) -> np.ndarray:
    """Apply an oriented IoU function to every predicted/ground-truth pair."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    if pred_boxes.ndim != 2 or pred_boxes.shape[1] < 7:
        raise ValueError(
            f"Expected pred_boxes of shape (N, 7+), got {pred_boxes.shape}"
        )
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] < 7:
        raise ValueError(f"Expected gt_boxes of shape (M, 7+), got {gt_boxes.shape}")

    ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)
    for pred_index, pred_box in enumerate(pred_boxes):
        for gt_index, gt_box in enumerate(gt_boxes):
            ious[pred_index, gt_index] = iou_function(pred_box, gt_box)
    return ious


class LiDARDetectionMetricsFactory(DetectionMetricsFactory):
    """Detection metrics for oriented LiDAR boxes.

    Boxes use the [x, y, z, dx, dy, dz, yaw] convention, where the center and
    yaw are expressed in the LiDAR frame.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        num_classes: Optional[int] = None,
        iou_mode: str = "3d",
    ):
        if iou_mode not in {"3d", "bev"}:
            raise ValueError("iou_mode must be '3d' or 'bev'.")
        super().__init__(iou_threshold=iou_threshold, num_classes=num_classes)
        self.iou_mode = iou_mode

    def compute_iou_matrix(
        self, pred_boxes: np.ndarray, gt_boxes: np.ndarray
    ) -> np.ndarray:
        """Compute the configured IoU matrix for predicted and GT boxes."""
        if self.iou_mode == "bev":
            return compute_bev_iou_matrix(pred_boxes, gt_boxes)
        return compute_3d_iou_matrix(pred_boxes, gt_boxes)

    def _match_predictions(
        self,
        gt_boxes: np.ndarray,
        gt_labels: List[int],
        pred_boxes: np.ndarray,
        pred_labels: List[int],
        pred_scores: List[float],
        iou_threshold: Optional[float] = None,
    ) -> Dict[int, List[Tuple[float, int]]]:
        """Match 3D predictions to ground truth using the configured IoU mode."""
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        results = defaultdict(list)
        used = set()
        ious = self.compute_iou_matrix(pred_boxes, gt_boxes)

        for index in np.argsort([-score for score in pred_scores]):
            pred_label = pred_labels[index]
            score = pred_scores[index]
            max_iou = 0
            matched_index = -1

            for gt_index, gt_label in enumerate(gt_labels):
                if gt_index in used or pred_label != gt_label:
                    continue
                iou = ious[index, gt_index]
                if iou > max_iou:
                    max_iou = iou
                    matched_index = gt_index

            if max_iou >= iou_threshold:
                results[pred_label].append((score, 1))
                used.add(matched_index)
            else:
                results[pred_label].append((score, 0))

        for gt_index, gt_label in enumerate(gt_labels):
            if gt_index not in used:
                results[gt_label].append((None, -1))

        return results

def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between pred and gt boxes using vectorized operations.

    :param pred_boxes: Predicted bounding boxes, shape (N, 4) in [x1, y1, x2, y2] format
    :type pred_boxes: np.ndarray
    :param gt_boxes: Ground truth bounding boxes, shape (M, 4) in [x1, y1, x2, y2] format
    :type gt_boxes: np.ndarray
    :return: IoU matrix with shape (N, M)
    :rtype: np.ndarray
    """
    assert (
        pred_boxes.ndim == 2 and pred_boxes.shape[1] == 4
    ), f"Expected pred_boxes of shape (N, 4), got {pred_boxes.shape}"
    assert (
        gt_boxes.ndim == 2 and gt_boxes.shape[1] == 4
    ), f"Expected gt_boxes of shape (M, 4), got {gt_boxes.shape}"

    pred_boxes = pred_boxes[:, np.newaxis, :]  # [N, 1, 4]
    gt_boxes = gt_boxes[np.newaxis, :, :]  # [1, M, 4]

    inter_x1 = np.maximum(pred_boxes[..., 0], gt_boxes[..., 0])  # [N, M]
    inter_y1 = np.maximum(pred_boxes[..., 1], gt_boxes[..., 1])  # [N, M]
    inter_x2 = np.minimum(pred_boxes[..., 2], gt_boxes[..., 2])  # [N, M]
    inter_y2 = np.minimum(pred_boxes[..., 3], gt_boxes[..., 3])  # [N, M]

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(
        0, inter_y2 - inter_y1
    )  # [N, M]

    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (
        pred_boxes[..., 3] - pred_boxes[..., 1]
    )  # [N, 1]
    gt_area = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (
        gt_boxes[..., 3] - gt_boxes[..., 1]
    )  # [1, M]

    union_area = pred_area + gt_area - inter_area  # [N, M]
    iou_matrix = inter_area / np.maximum(union_area, 1e-8)  # avoiding zero division

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
    union = float(boxAArea + boxBArea - interArea)
    iou = interArea / union if union > 0 else 0.0
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
