import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class DetectionMetricsFactory:
    def __init__(self, iou_threshold: float = 0.5, num_classes: Optional[int] = None):
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.results = defaultdict(list)  # stores detection results per class

    def update(self, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores):
        """
        Add a batch of predictions and ground truths.

        :param gt_boxes: List[ndarray], shape (num_gt, 4)
        :param gt_labels: List[int]
        :param pred_boxes: List[ndarray], shape (num_pred, 4)
        :param pred_labels: List[int]
        :param pred_scores: List[float]
        """

        # Convert torch tensors to numpy
        if hasattr(gt_boxes, 'detach'): gt_boxes = gt_boxes.detach().cpu().numpy()
        if hasattr(gt_labels, 'detach'): gt_labels = gt_labels.detach().cpu().numpy()
        if hasattr(pred_boxes, 'detach'): pred_boxes = pred_boxes.detach().cpu().numpy()
        if hasattr(pred_labels, 'detach'): pred_labels = pred_labels.detach().cpu().numpy()
        if hasattr(pred_scores, 'detach'): pred_scores = pred_scores.detach().cpu().numpy()

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
    ) -> Dict[int, List[Tuple[float, int]]]:
        """
        Match predictions to ground truth and return per-class TP/FP flags with scores.

        Returns:
            Dict[label_id, List[(score, tp_or_fp)]]
        """

        results = defaultdict(list)
        used = set()

        ious = compute_iou_matrix(pred_boxes, gt_boxes)  # shape: (num_preds, num_gts)

        for i, (p_box, p_label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            max_iou = 0
            max_j = -1

            for j, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
                if j in used or p_label != g_label:
                    continue
                iou = ious[i, j]
                if iou > max_iou:
                    max_iou = iou
                    max_j = j

            if max_iou >= self.iou_threshold:
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
        """
        Compute per-class precision, recall, AP, and mAP.

        Returns:
            Dict[class_id, Dict[str, float]], plus an entry for mAP under key -1
        """
        metrics = {}
        ap_values = []

        for label, detections in self.results.items():
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


    def get_metrics_dataframe(self, ontology: dict) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.

        :param ontology: Mapping from class name → { "idx": int }
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

        df = pd.DataFrame(metrics_dict)
        return df.T  # metrics as rows, classes as columns (with mean)


def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between pred and gt boxes.
    """
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pb, gb)
    return iou_matrix

def compute_iou(boxA, boxB):
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
    tps = np.array(tps, dtype=np.float32)
    fps = np.array(fps, dtype=np.float32)

    tp_cumsum = np.cumsum(tps)
    fp_cumsum = np.cumsum(fps)

    if tp_cumsum.size:
        denom = tp_cumsum[-1] + fn
        recalls = tp_cumsum / denom
    else:
        recalls = []
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6) if tp_cumsum.size else []

    # VOC-style 11-point interpolation
    ap = 0
    for r in np.linspace(0, 1, 11):
        p = [p for p, rc in zip(precisions, recalls) if rc >= r]
        ap += max(p) if p else 0
    ap /= 11.0

    return ap, precisions, recalls
