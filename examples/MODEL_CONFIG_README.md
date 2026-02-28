# Object Detection Model Configuration Guide

This directory contains example configuration files for object detection models in PerceptionMetrics.

## Configuration Parameters

### Required Parameters

- **`model_format`**: Format of the model output (`"yolo"` or `"torchvision"`)
- **`normalization`**: Input normalization parameters
  - `mean`: RGB mean values for normalization
  - `std`: RGB standard deviation values for normalization

### Optional Parameters

- **`resize`**: Input image resizing (default: 640x640)
  - `width`: Target width in pixels
  - `height`: Target height in pixels

- **`confidence_threshold`**: Confidence threshold for precision/recall/confusion matrix computation
  - If specified: Use this threshold (e.g., `0.25`)
  - If omitted: Automatically select the threshold that maximizes F1 score
  - **Note**: This does NOT affect mAP or PR curve computation (those use all predictions)

- **`nms_threshold`**: IoU threshold for Non-Maximum Suppression (default: `0.45`)
  - Controls duplicate detection removal
  - Lower values = more aggressive NMS (fewer overlapping boxes)

- **`iou_threshold`**: IoU threshold for matching predictions to ground truth (default: `0.5`)
  - Used in mAP, precision, recall, and confusion matrix computation
  - Standard value is 0.5 for PASCAL VOC-style evaluation

- **`batch_size`**: Number of images to process simultaneously (default: `1`)

- **`num_workers`**: Number of data loading workers for PyTorch DataLoader (default: `4`)

- **`device`**: Computation device (`"cuda"`, `"mps"`, or `"cpu"`)

- **`evaluation_step`**: How often to report intermediate metrics during evaluation (in number of images)
  - Useful for progress tracking on large datasets
  - Set to `0` or omit to disable intermediate reporting

## Important: Confidence Threshold Behavior

PerceptionMetrics follows the [Ultralytics YOLO evaluation methodology](https://docs.ultralytics.com/guides/yolo-performance-metrics/):

### For mAP and Precision-Recall Curves
- **No confidence filtering is applied**
- All predictions are kept regardless of the `confidence_threshold` setting
- This allows proper evaluation across the full confidence range

### For Precision, Recall, and Confusion Matrix
- The `confidence_threshold` parameter IS applied
- **Auto-Selection**: Omit `confidence_threshold` to use the value that maximizes F1 score
- **Manual**: Specify a value (e.g., `0.25`) to use a fixed threshold

See the [Detection Evaluation Methodology](https://jderobot.github.io/PerceptionMetrics/detection_evaluation/) documentation for detailed explanation.

## Example Configurations

### YOLO Model (with Auto-Selected Threshold)

```json
{
  "model_format": "yolo",
  "normalization": {
    "mean": [0.0, 0.0, 0.0],
    "std": [255.0, 255.0, 255.0]
  },
  "resize": {
    "width": 640,
    "height": 640
  },
  "nms_threshold": 0.45,
  "iou_threshold": 0.5,
  "batch_size": 8,
  "device": "cuda"
}
```
*Note: `confidence_threshold` omitted - will automatically find optimal value*

### YOLO Model (with Fixed Threshold)

```json
{
  "model_format": "yolo",
  "normalization": {
    "mean": [0.0, 0.0, 0.0],
    "std": [255.0, 255.0, 255.0]
  },
  "resize": {
    "width": 640,
    "height": 640
  },
  "confidence_threshold": 0.25,
  "nms_threshold": 0.45,
  "iou_threshold": 0.5,
  "batch_size": 8,
  "device": "cuda"
}
```
*Note: Uses fixed confidence threshold of 0.25*

### TorchVision Model (Faster R-CNN, etc.)

```json
{
  "model_format": "torchvision",
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "resize": {
    "width": 800,
    "height": 600
  },
  "confidence_threshold": 0.5,
  "iou_threshold": 0.5,
  "batch_size": 4,
  "device": "cuda"
}
```

## Accessing Optimal Threshold

When `confidence_threshold` is omitted, you can retrieve the auto-selected value:

```python
from perceptionmetrics.models import TorchImageDetectionModel

model = TorchImageDetectionModel(
    model="model.pt",
    model_cfg="config.json",  # without confidence_threshold
    ontology_fname="ontology.json"
)

results = model.evaluate(dataset, split="test")

# Get optimal threshold info
optimal_info = model.metrics_factory.get_optimal_conf_threshold()
print(f"Optimal confidence: {optimal_info['threshold']:.3f}")
print(f"F1 score: {optimal_info['f1_score']:.3f}")
```

## Further Reading

- [PerceptionMetrics Detection Evaluation Methodology](https://jderobot.github.io/PerceptionMetrics/detection_evaluation/)
- [Ultralytics YOLO Performance Metrics Guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [COCO Detection Evaluation](https://cocodataset.org/#detection-eval)
