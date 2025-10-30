---
layout: home
title: GUI
permalink: /v2/gui/

sidebar:
  nav: "main_v2"
---

# DetectionMetrics GUI

DetectionMetrics includes a **Streamlit-based GUI** that provides an intuitive interface for image detection model evaluation, dataset exploration, and real-time inference. The GUI makes it easy to work with detection models without writing code.

## Getting Started

### Installation
The GUI is included with DetectionMetrics and requires no additional installation steps beyond the standard setup.

### Launching the GUI
```bash
# From the project root directory
streamlit run app.py
```

The GUI will open in your default web browser, typically at `http://localhost:8501`.

## GUI Overview

The GUI consists of three main tabs, each designed for specific tasks:

### 📁 Dataset Viewer Tab

The Dataset Viewer allows you to explore and visualize your datasets before running evaluations.

#### Features:
- **Dataset Loading**: Load datasets from the sidebar configuration
- **Image Navigation**: Browse through dataset images with intuitive controls
- **Annotation Visualization**: View ground truth annotations overlaid on images
- **Split Selection**: Switch between train/val/test splits
- **Dataset Information**: Display dataset statistics and metadata

#### Supported Formats:
- **COCO**: Standard COCO format for object detection
- **YOLO**: Ultralytics YOLO dataset format

#### Usage:
1. Configure dataset path and type in the sidebar
2. Select the desired split (train/val)
3. Navigate through images using the controls
4. You can also click on the search button to directly navigate to any desired image.
5. View annotations and dataset information

### 🔍 Inference Tab

The Inference tab provides real-time model inference capabilities for testing and visualization.

#### Features:
- **Model Loading**: Load trained models through the sidebar
- **Real-time Inference**: Run inference on individual images
- **Result Visualization**: Display detection results with bounding boxes
- **Parameter Adjustment**: Modify confidence thresholds and other parameters
- **Custom Image Upload**: Test on your own images

#### Supported Models:
- **PyTorch**: TorchScript (.pt) and native (.pth) models (detection only)

#### Usage:
1. Load a model using the sidebar configuration
2. Upload an image or select from the dataset
3. Adjust inference parameters as needed
4. Run inference and view results
5. Analyze detection confidence and bounding boxes
6. Download the predictions.

### 📊 Evaluator Tab

The Evaluator tab provides comprehensive model evaluation capabilities with real-time progress tracking.

#### Features:
- **Batch Evaluation**: Evaluate models on entire datasets
- **Progress Tracking**: Real-time progress updates during evaluation
- **Metrics Display**: View comprehensive evaluation metrics
- **Result Export**: Save detailed results and predictions

#### Evaluation Metrics:
- mAP (including COCO-style mAP@0.5:0.95:0.05), AUC (Area Under Curve), Precision, Recall, F1-Score

#### Usage:
1. Ensure both model and dataset are loaded in the input side bar.
2. Configure evaluation parameters
3. Optionally upload ontology translation file
4. Run evaluation and monitor progress live
5. Review results and download results

## Sidebar Configuration

The sidebar provides centralized configuration for all GUI components:

### Dataset Inputs
- **Type**: Select dataset format (COCO)
- **Split**: Choose dataset split (train, val)
- **Path**: Set dataset directory path
- **Browse**: Use file browser to select dataset location

### Model Inputs
- **Model File**: Upload trained model file
- **Ontology File**: Upload class definition JSON
- **Configuration**: Choose between manual settings or config file upload

#### Manual Configuration Options:
- **Confidence Threshold**: Minimum confidence for detections
- **NMS Threshold**: Non-maximum suppression threshold
- **Max Detections**: Maximum detections per image
- **Device**: CPU, CUDA, or MPS
- **Batch Size**: Inference batch size
- **Evaluation Step**: Live progress update frequency

## Configuration Files

### Model Configuration JSON
```json
{
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "confidence_threshold": 0.5,
    "nms_threshold": 0.5,
    "max_detections_per_image": 100,
    "batch_size": 1,
    "device": "cpu",
    "evaluation_step": 10,
    "model_format": "coco"
}
```

### Ontology JSON
```json
{
    "person": {"idx": 1, "rgb": [255, 0, 0]},
    "car": {"idx": 2, "rgb": [0, 255, 0]},
    "bicycle": {"idx": 3, "rgb": [0, 0, 255]}
}
```

## Tips and Best Practices

### Performance Optimization
- Use GPU acceleration when available (CUDA/MPS)
- Adjust batch size based on available memory
- Use evaluation_step to control progress update frequency

### Dataset Preparation
- Ensure COCO datasets follow standard directory structure
- Verify ontology files match model output classes
- Check image and annotation file paths

### Model Compatibility
- Test models with sample images in inference tab before full evaluation
- Verify model input/output formats match expectations
- Use appropriate confidence thresholds for your use case

## Troubleshooting

### Common Issues
- **Model Loading Errors**: Check model format and file integrity
- **Dataset Loading Issues**: Verify dataset structure and file paths
- **Memory Issues**: Reduce batch size or use CPU inference
- **GUI Performance**: Adjust evaluation_step or use smaller datasets
