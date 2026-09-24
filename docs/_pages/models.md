---
layout: home
title: Models
permalink: /models/

sidebar:
  nav: "main"
---

This page summarizes the model wrappers available in PerceptionMetrics and the configuration files they expect. Model wrappers normalize framework-specific inference into the common `PerceptionModel` API used by evaluation, prediction export, and the GUI.

## Support Matrix

| Model wrapper | Task | Modality | Registry key | Status |
| --- | --- | --- | --- | --- |
| `TorchImageSegmentationModel` | Segmentation | Image | `torch_image_segmentation` | Registered |
| `TorchLiDARSegmentationModel` | Segmentation | LiDAR | `torch_lidar_segmentation` | Registered |
| `TorchImageDetectionModel` | Object detection | Image | `torch_image_detection` | Registered |
| `TensorflowImageSegmentationModel` | Segmentation | Image | `tensorflow_image_segmentation` | Registered |




## Model Ontology

The model ontology maps output class names to model output indices. It uses the same basic shape as dataset ontologies:

```json
{
  "road": {
    "idx": 0,
    "rgb": [128, 64, 128]
  },
  "vegetation": {
    "idx": 1,
    "rgb": [107, 142, 35]
  }
}
```

If model and dataset class spaces differ, provide an ontology translation during evaluation. The translation file is interpreted by the evaluation code to build a lookup table between dataset labels and model labels.

## Image Segmentation Config

PyTorch and TensorFlow image segmentation wrappers share a similar JSON configuration:

```json
{
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "resize": {
    "width": 1024,
    "height": 512
  },
  "crop": {
    "width": 1024,
    "height": 512
  },
  "batch_size": 1,
  "num_workers": 1,
  "ignored_classes": ["unlabeled"]
}
```

Fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `normalization` | No | Mean and standard deviation applied after converting images to float |
| `resize` | No | Resize input image and label before inference/evaluation |
| `crop` | No | Center crop input image and label |
| `batch_size` | No | Evaluation batch size; defaults to `1` |
| `num_workers` | No | PyTorch dataloader workers; defaults vary by wrapper |
| `ignored_classes` | No | Dataset class names to exclude from metric updates |
| `keep_aspect` | TensorFlow only | Resize while preserving aspect ratio before center crop |

Expected image input shape:

| Framework | Input shape | Output shape |
| --- | --- | --- |
| PyTorch | `(batch, channels, height, width)` | `(batch, classes, height, width)` |
| TensorFlow | `(batch, height, width, channels)` | `(batch, height, width, classes)` |

For PyTorch image segmentation, the wrapper accepts either a TorchScript file, a serialized PyTorch module, or an already loaded `torch.nn.Module` from Python.

For TensorFlow image segmentation, the wrapper accepts either a SavedModel directory or a loaded TensorFlow/Keras model from Python.

## LiDAR Segmentation Config

`TorchLiDARSegmentationModel` supports several LiDAR model utility formats through `model_cfg["model_format"]`:

```json
{
  "model_format": "mmdet3d",
  "n_feats": 4,
  "batch_size": 1
}
```

Supported LiDAR utility formats in this branch:

| `model_format` | Utility module | Notes |
| --- | --- | --- |
| `o3d_randlanet` | `perceptionmetrics.models.utils.o3d` | Open3D-ML RandLA-Net style inputs |
| `o3d_kpconv` | `perceptionmetrics.models.utils.o3d` | Open3D-ML KPConv style inputs |
| `mmdet3d` | `perceptionmetrics.models.utils.mmdet3d` | MMDetection3D style point segmentation |
| `sphereformer` | `perceptionmetrics.models.utils.sphereformer` | Requires the SphereFormer-specific environment |
| `lsk3dnet` | `perceptionmetrics.models.utils.lsk3dnet` | Requires the LSK3DNet-specific environment |

Common fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `model_format` | Yes | LiDAR utility format |
| `n_feats` | Usually | Number of point features, commonly `3` or `4` |
| `batch_size` | No | Evaluation batch size |
| `ignored_classes` | No | Dataset class names to exclude from metrics |

Open3D-ML style configs may include sampler and neighborhood parameters such as `sampler`, `num_points`, `grid_size`, `num_neighbors`, and `sub_sampling_ratio`.

SphereFormer style configs may include:

```json
{
  "model_format": "sphereformer",
  "n_feats": 4,
  "voxel_size": [0.05, 0.05, 0.05],
  "voxel_max": 120000,
  "pc_range": [[-22, -17, -4], [30, 18, 13]],
  "xyz_norm": false
}
```

LSK3DNet style configs may include:

```json
{
  "model_format": "lsk3dnet",
  "n_feats": 4,
  "min_volume_space": [-120, -120, -6],
  "max_volume_space": [120, 120, 11]
}
```

Additional environment setup for MMDetection3D, SphereFormer, and LSK3DNet is documented in `additional_envs/INSTRUCTIONS.md`.

## Image Detection Config

`TorchImageDetectionModel` supports TorchVision-style detection outputs and TorchScript-exported YOLO-style outputs. Choose the post-processing path with `model_cfg["model_format"]`.

```json
{
  "model_format": "torchvision",
  "resize": {
    "min_side": 800,
    "max_side": 1333
  },
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "confidence_threshold": 0.5,
  "nms_threshold": 0.3,
  "iou_threshold": 0.5,
  "batch_size": 1,
  "num_workers": 0,
  "evaluation_step": 25
}
```

Fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `model_format` | No | `torchvision` by default; use `yolo` for YOLO post-processing |
| `resize` | No | Either fixed `width`/`height`, or `min_side` with optional `max_side` |
| `crop` | No | Center crop after resize |
| `normalization` | No | Mean and standard deviation applied to input tensors |
| `confidence_threshold` | No | Minimum score for retained detections |
| `nms_threshold` | YOLO only | Non-maximum suppression threshold |
| `iou_threshold` | No | IoU threshold used by detection metrics |
| `batch_size` | No | Evaluation batch size |
| `num_workers` | No | Dataloader workers |
| `evaluation_step` | No | Frequency for intermediate metric updates in GUI/evaluation callbacks |

Detection predictions are expected after post-processing as dictionaries with:

```text
boxes:  [N, 4] in XYXY format
labels: [N]
scores: [N]
```

## Python API Examples

```python
from perceptionmetrics.models.torch_segmentation import TorchImageSegmentationModel

model = TorchImageSegmentationModel(
    model="/path/to/model.pt",
    model_cfg="/path/to/model_cfg.json",
    ontology_fname="/path/to/model_ontology.json",
)
```

```python
from perceptionmetrics.models.torch_detection import TorchImageDetectionModel

model = TorchImageDetectionModel(
    model="/path/to/detector.pt",
    model_cfg="/path/to/detection_cfg.json",
    ontology_fname="/path/to/model_ontology.json",
)
```

```python
from perceptionmetrics.models.tf_segmentation import TensorflowImageSegmentationModel

model = TensorflowImageSegmentationModel(
    model="/path/to/saved_model",
    model_cfg="/path/to/model_cfg.json",
    ontology_fname="/path/to/model_ontology.json",
)
```
