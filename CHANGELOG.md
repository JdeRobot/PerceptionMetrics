# Changelog

All notable changes to PerceptionMetrics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed - Object Detection Evaluation (Aligned with Ultralytics)

#### mAP and Precision-Recall Curves
- **BREAKING CHANGE**: mAP and PR curves now computed WITHOUT confidence threshold filtering
- All predicted boxes are kept for evaluation, regardless of confidence score
- This aligns with industry-standard evaluation practices and Ultralytics YOLO
- **Impact**: mAP values may differ from previous versions but are now more accurate

#### Confidence Threshold Handling
- **NEW**: Separate confidence threshold handling for different metrics:
  - mAP and PR curves: No filtering (all predictions)
  - Precision, Recall, Confusion Matrix: Filtered at specified or optimal threshold
- **NEW**: Auto-selection of optimal confidence threshold
  - When `confidence_threshold` is omitted from config, automatically selects value that maximizes F1 score
  - Optimal threshold and F1 score are reported
- Model config now supports optional `confidence_threshold` parameter

#### Confusion Matrix Enhancements
- **NEW**: Confusion matrices now include implicit background class (following Ultralytics convention)
  - Unmatched predictions (FP) counted as `predicted_class` vs `background`
  - Unmatched ground truth (FN) counted as `background` vs `true_class`
- Matrix shape is now (N+1) × (N+1) for N classes when `include_background=True`
- Provides complete view of model errors including missed detections

#### New API Methods
- `DetectionMetricsFactory.get_confusion_matrix()`: Compute confusion matrix with background class support
- `DetectionMetricsFactory.get_optimal_conf_threshold()`: Get optimal threshold that maximizes F1 score
- `DetectionMetricsFactory._find_optimal_confidence_threshold()`: Internal method for threshold optimization

#### Model Configuration
- `confidence_threshold` in model config now optional (defaults to auto-selection)
- When specified, used only for precision/recall/confusion matrix (not mAP/PR curves)
- Postprocessing functions updated to accept `confidence_threshold=0` for keeping all predictions

### Added

#### Documentation
- New comprehensive [Detection Evaluation Methodology](docs/_pages/detection_evaluation.md) guide
- Detailed explanation of confidence threshold strategy
- Confusion matrix with background class documentation
- Example model configurations in `examples/MODEL_CONFIG_README.md`
- Updated [Compatibility](docs/_pages/compatibility.md) page with evaluation methodology section

#### Examples
- `examples/yolo_model_config_example.json`: Example YOLO model configuration
- `examples/MODEL_CONFIG_README.md`: Comprehensive guide for model configuration parameters

### Fixed
- Detection evaluation now correctly aligns with Ultralytics YOLO methodology
- mAP computation no longer incorrectly filters predictions by confidence threshold
- Confusion matrices now account for all prediction errors (including missed detections)

### Migration Guide

If you're upgrading from a previous version:

1. **mAP Values May Change**: Your mAP values may be different (likely higher) because all predictions are now included. This is the correct behavior.

2. **Confusion Matrix Shape**: If you're processing confusion matrices, note they now have an extra background class dimension by default.

3. **Optimal Threshold**: Consider removing `confidence_threshold` from your model configs to use auto-selected optimal values.

4. **Code Updates**: If you access confusion matrices directly:
   ```python
   # Old code (may fail if background class included)
   cm = metrics_factory.get_confusion_matrix()
   
   # New code (explicitly control background class)
   cm = metrics_factory.get_confusion_matrix(include_background=True)  # (N+1, N+1)
   # or
   cm = metrics_factory.get_confusion_matrix(include_background=False)  # (N, N)
   ```

### References
- [Ultralytics YOLO Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [COCO Detection Evaluation](https://cocodataset.org/#detection-eval)

---

## [3.0.1] - 2025-01-XX

Previous releases... (to be added)
