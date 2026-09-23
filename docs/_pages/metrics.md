---
layout: home
title: Metrics
permalink: /metrics/

sidebar:
  nav: "main"
---

This page summarizes the metrics computed by PerceptionMetrics for segmentation, object detection, and model profiling. The implementation lives in `perceptionmetrics.utils.segmentation_metrics` and `perceptionmetrics.utils.detection_metrics`.

## Segmentation Metrics

Segmentation metrics are accumulated with `SegmentationMetricsFactory`. The factory stores a confusion matrix of shape `(n_classes, n_classes)` and updates it from integer prediction and ground-truth arrays.

Supported metrics:

| Metric | Meaning |
| --- | --- |
| `tp` | True positives |
| `fp` | False positives |
| `fn` | False negatives |
| `tn` | True negatives |
| `precision` | `TP / (TP + FP)` |
| `recall` | `TP / (TP + FN)` |
| `accuracy` | `(TP + TN) / (TP + FP + FN + TN)` |
| `f1_score` | Harmonic mean of precision and recall |
| `iou` | Intersection over Union, `TP / (TP + FP + FN)` |
| `dice_score` | Dice coefficient, `2TP / (2TP + FP + FN)` |

The metric factory supports per-class values and global values. Global metrics can be averaged as:

| Average | Meaning |
| --- | --- |
| `macro` | Mean of per-class metric values, ignoring NaNs |
| `micro` | Metric computed from globally summed counts |
| `weighted` | Weighted sum of per-class values using provided weights |
| `normalized_weighted` | Weighted mean normalized by the sum of weights |

## Segmentation Output Table

`get_metrics_dataframe()` returns a DataFrame with:

- one column per class
- `macro` and `micro` columns for averaged metrics
- metric rows such as `precision`, `recall`, `iou`, and `dice_score`
- confusion-matrix rows using class names

Example shape:

```text
                 road   vegetation   sky   macro   micro
precision        ...    ...          ...   ...     ...
recall           ...    ...          ...   ...     ...
iou              ...    ...          ...   ...     ...
dice_score       ...    ...          ...   ...     ...
road             ...    ...          ...
vegetation       ...    ...          ...
sky              ...    ...          ...
```

The class-name rows at the bottom represent the confusion matrix. For those rows, each column stores the count of predictions assigned to that column class for samples whose ground-truth class is the row class.

## Ignored Labels

Segmentation evaluation can ignore labels by passing a valid mask into the metric update. Model wrappers build this mask from `ignored_classes` in the model configuration. Ignored pixels or points are excluded before the confusion matrix is updated.

## Detection Metrics

Object detection metrics are accumulated with `DetectionMetricsFactory`. The factory receives ground-truth boxes, predicted boxes, labels, and confidence scores.

Predictions and ground truth boxes are expected in `[x1, y1, x2, y2]` format.

Supported metrics:

| Metric | Meaning |
| --- | --- |
| `AP` | Average Precision per class using VOC-style 11-point interpolation |
| `Precision` | Final precision value from the precision-recall curve |
| `Recall` | Final recall value from the precision-recall curve |
| `TP` | Number of matched true-positive detections |
| `FP` | Number of unmatched predictions |
| `FN` | Number of missed ground-truth objects |
| `mAP@[0.5:0.95]` | COCO-style mean AP over IoU thresholds from `0.5` to `0.95` |
| `AUC-PR` | Area under the overall precision-recall curve |

Detection matching uses an IoU threshold. A prediction is counted as a true positive when:

- it has the same class as an unmatched ground-truth box
- its IoU with that ground-truth box is greater than or equal to the threshold

Unmatched predictions become false positives. Unmatched ground-truth boxes become false negatives.

## Detection Output Table

`DetectionMetricsFactory.get_metrics_dataframe()` returns a DataFrame with metric names as rows and class names as columns.

Example shape:

```text
                    person   vehicle   mean
AP                  ...      ...       ...
Precision           ...      ...       ...
Recall              ...      ...       ...
TP                  ...      ...       ...
FP                  ...      ...       ...
FN                  ...      ...       ...
mAP@[0.5:0.95]      NaN      NaN       ...
AUC-PR              NaN      NaN       ...
```

The `mean` column stores the mean value across valid class values. COCO-style mAP and AUC-PR are stored only in the `mean` column.

## Profiling Metrics

PerceptionMetrics also reports model profiling values through model `get_computational_cost()` methods. These are not dataset-quality metrics, but they are useful when comparing deployment cost.

Common profiling fields:

| Field | Meaning |
| --- | --- |
| `input_shape` | Shape of the dummy input used for profiling |
| `n_params` | Number of model parameters |
| `size_mb` | Model file size in megabytes when a model filename is available |
| `inference_time_s` | Mean inference time in seconds over repeated runs |

Image profiling uses a dummy image tensor. LiDAR profiling uses a dummy point cloud generated from a point-cloud range, number of points, and optional intensity channel.

## Python API Examples

```python
import numpy as np
from perceptionmetrics.utils.segmentation_metrics import SegmentationMetricsFactory

metrics = SegmentationMetricsFactory(n_classes=3)
metrics.update(
    pred=np.array([[0, 1], [1, 2]], dtype=np.int64),
    gt=np.array([[0, 1], [2, 2]], dtype=np.int64),
)

iou_per_class = metrics.get_iou(per_class=True)
```

```python
from perceptionmetrics.utils.detection_metrics import DetectionMetricsFactory

metrics = DetectionMetricsFactory(iou_threshold=0.5, num_classes=2)
metrics.update(
    gt_boxes=[[10, 10, 50, 50]],
    gt_labels=[0],
    pred_boxes=[[12, 12, 48, 48]],
    pred_labels=[0],
    pred_scores=[0.9],
)

results = metrics.compute_metrics()
```
