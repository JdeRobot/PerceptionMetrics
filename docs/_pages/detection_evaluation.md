---
layout: home
title: Detection Evaluation Methodology
permalink: /detection_evaluation/

sidebar:
  nav: "main"
---

# Object Detection Evaluation Methodology

PerceptionMetrics follows industry-standard evaluation practices for object detection, aligned with [Ultralytics YOLO](https://docs.ultralytics.com/guides/yolo-performance-metrics/) to ensure consistency and comparability with widely-used frameworks.

## Overview

Detection evaluation involves different metrics that serve different purposes. Understanding when and how confidence thresholds are applied is crucial for proper interpretation of results.

## Confidence Threshold Strategy

### For mAP and Precision-Recall Curves

**No confidence threshold is applied during evaluation.**

- All predicted bounding boxes are kept, regardless of their confidence score
- Predictions are ranked by confidence for threshold-free evaluation
- This allows computing metrics across the full confidence range (0 to 1)
- Ensures proper ranking-based evaluation as intended by mAP definition

**Why?** mAP (mean Average Precision) and PR curves are designed to evaluate model performance across all possible confidence thresholds. Filtering predictions beforehand would artificially limit this evaluation and produce incorrect results.

### For Precision, Recall, and Confusion Matrix

**A specific confidence threshold is applied.**

You have two options:

1. **User-Defined Threshold**: Specify `confidence_threshold` in your model configuration JSON
2. **Auto-Selected Threshold**: Omit `confidence_threshold` to automatically use the value that maximizes F1 score

When auto-selection is used:
- The optimal threshold is computed from the evaluation data
- Both the threshold value and the corresponding F1 score are reported
- This represents the best possible performance for threshold-dependent metrics

**Example Configuration:**

```json
{
  "model_format": "yolo",
  "confidence_threshold": 0.25,  // Optional: omit for auto-selection
  "nms_threshold": 0.45,
  "iou_threshold": 0.5,
  "batch_size": 8
}
```

## Confusion Matrix with Background Class

Following Ultralytics convention, PerceptionMetrics includes an implicit **background class** in confusion matrices to account for all prediction errors:

### How It Works

1. **True Positives (TP)**: Predictions correctly matched to ground truth boxes (IoU ≥ threshold, same class)
   - Counted in confusion matrix: `matrix[true_class, predicted_class]`

2. **False Positives (FP)**: Predictions with no matching ground truth
   - Counted as: `matrix[background, predicted_class]`
   - Interpretation: "Model predicted `predicted_class`, but there was no object (background)"

3. **False Negatives (FN)**: Ground truth boxes with no matching prediction
   - Counted as: `matrix[true_class, background]`
   - Interpretation: "Object was `true_class`, but model predicted nothing (background)"

### Matrix Structure

For a dataset with N classes, the confusion matrix has shape **(N+1) × (N+1)**:

```
                 Predicted Classes
                 C0   C1   C2   ... Background
True Classes   
C0              TP   FP   FP   ... FN
C1              FP   TP   FP   ... FN  
C2              FP   FP   TP   ... FN
...
Background      FP   FP   FP   ... --
```

Where:
- Diagonal elements (except background): True Positives
- Off-diagonal elements (excluding background row/column): Wrong class predictions
- Background row: False Positives (predicted something, but nothing was there)
- Background column: False Negatives (missed detections)

## Matching Algorithm

Predictions are matched to ground truth using:

1. **IoU Threshold**: Default 0.5 (configurable via `iou_threshold` in model config)
2. **Class Matching**: Prediction and ground truth must have the same class label
3. **Greedy Matching**: Each prediction matched to best available ground truth (highest IoU)
4. **One-to-One**: Each ground truth can only be matched once

## Metrics Computed

### Threshold-Free Metrics (No Confidence Filtering)

- **mAP (mean Average Precision)**: Mean AP across all classes at specified IoU threshold
- **mAP@[0.5:0.95]**: COCO-style mAP averaged over IoU thresholds from 0.5 to 0.95 (step 0.05)
- **Precision-Recall Curves**: Full curves showing performance trade-offs
- **AUC-PR**: Area under the precision-recall curve

### Threshold-Dependent Metrics (With Confidence Filtering)

Applied at user-defined or F1-maximizing threshold:

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Confusion Matrix**: With background class as described above
- **Per-Class Metrics**: Individual precision, recall, and F1 for each class

## Implementation Details

### Model Inference

During evaluation, PerceptionMetrics:

1. Performs inference with **confidence_threshold = 0** to keep all predictions
2. Applies Non-Maximum Suppression (NMS) to remove duplicate detections
3. Stores all predictions with their confidence scores
4. Computes mAP/PR curves on the full prediction set
5. Filters by confidence threshold only for precision/recall/confusion matrix computation

### Code Example

```python
from perceptionmetrics.datasets import ImageDetectionDataset
from perceptionmetrics.models import TorchImageDetectionModel

# Load dataset
dataset = ImageDetectionDataset(
    dataset_format="yolo",
    dataset_dir="/path/to/dataset",
    ontology_fname="ontology.json"
)

# Load model
model = TorchImageDetectionModel(
    model="/path/to/model.pt",
    model_cfg="model_config.json",  # confidence_threshold optional
    ontology_fname="model_ontology.json"
)

# Evaluate
results = model.evaluate(dataset, split="test")

# Access metrics
map_50 = results.loc["AP", "mean"]  # mAP@0.5
map_coco = results.loc["mAP@[0.5:0.95]", "mean"]  # COCO mAP

# Get optimal confidence threshold (if auto-selected)
optimal_conf_info = model.metrics_factory.get_optimal_conf_threshold()
print(f"Optimal confidence: {optimal_conf_info['threshold']:.3f}")
print(f"F1 score at optimal conf: {optimal_conf_info['f1_score']:.3f}")

# Get confusion matrix with background class
conf_matrix = model.metrics_factory.get_confusion_matrix(include_background=True)
```

## References

This evaluation methodology is aligned with:

- [Ultralytics YOLO Performance Metrics Guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [COCO Detection Evaluation](https://cocodataset.org/#detection-eval)
- [PASCAL VOC Detection Evaluation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000)

## Key Differences from Previous Versions

If you're migrating from an earlier version of PerceptionMetrics (DetectionMetrics):

1. **mAP Computation**: Now computed on ALL predictions (no conf filtering), aligning with standard practice
2. **Confusion Matrix**: Now includes background class for complete error analysis
3. **Optimal Threshold**: New auto-selection feature for confidence threshold
4. **Consistent with Ultralytics**: Results now match Ultralytics YOLO evaluation

These changes ensure more accurate and comparable detection evaluation results.
