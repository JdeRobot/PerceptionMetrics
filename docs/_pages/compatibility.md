---
layout: home
title: Compatibility
permalink: /compatibility/

sidebar:
  nav: "main"
---

## Image semantic segmentation
- Datasets:
    - **[RUGD](http://rugd.vision/)**
    - **[Rellis3D](https://www.unmannedlab.org/research/RELLIS-3D)**
    - **[GOOSE](https://goose-dataset.de/)**
    - **[WildScenes](https://csiro-robotics.github.io/WildScenes/)**
    - **Custom GAIA format**: *Parquet* file containing samples and labels relative paths and a JSON file with the dataset ontology.
    - **Generic**: simply assumes a different directory per split, different suffixes for samples and labels, and a JSON file containing the dataset ontology.
- Models:
    - **PyTorch ([TorchScript](https://pytorch.org/docs/stable/jit.html) compiled format and native modules)**:
        - Input shape: `(batch, channels, height, width)`
        - Output shape: `(batch, classes, height, width)`
    - **Tensorflow ([SavedModel](https://www.tensorflow.org/guide/saved_mode`) compiled format and native Tensorflow/Keras modules)**:
        - Input shape: `(batch, height, width, channels)`
        - Output shape: `(batch, height, width, classes)`
        - JSON configuration file format:
    - **ONNX**: coming soon

    Each model must be coupled with a JSON configuration file:

    ```json
    {
        "normalization": {
            "mean": [<r>, <g>, <b>],
            "std": [<r>, <g>, <b>]
        },
        "resize": {  # optional
            "width": <px>,
            "height": <px>
        },
        "crop": {  # optional
            "width": <px>,
            "height": <px>
        },
        "batch_size": <n>
    }
    ```

- Metrics:
    - Intersection over Union (IoU), Accuracy
- Computational cost:
    - Number of parameters, average inference time, model size

## LiDAR semantic segmentation
- Datasets:
    - **[Rellis3D](https://www.unmannedlab.org/research/RELLIS-3D)**
    - **[GOOSE](https://goose-dataset.de/)**
    - **[WildScenes](https://csiro-robotics.github.io/WildScenes/)**
    - **Custom GAIA format**: *Parquet* file containing samples and labels relative paths and a JSON file with the dataset ontology.
    - **Generic**: simply assumes a different directory per split, different suffixes for samples and labels, and a JSON file containing the dataset ontology.
- Models:
    - **PyTorch ([TorchScript](https://pytorch.org/docs/stable/jit.html) compiled format and native modules)**. As of now, we have tested <a href="https://github.com/isl-org/Open3D-ML">Open3D-ML</a>, <a href="https://github.com/open-mmlab/mmdetection3d">mmdetection3d</a>, <a href="https://github.com/dvlab-research/SphereFormer">SphereFormer</a>, and <a href="https://github.com/FengZicai/LSK3DNet">LSK3DNet</a> models.
        - Input shape: defined by the `input_format` tag.
        - Output shape: `(num_points)`
        - JSON configuration file format examples (different depending on the model):

        ```json
        {
            "model_format": <"o3d_randlanet" | "o3d_kpconv" | "mmdet3d" | "sphereformer" | "lsk3dnet">,
            "n_feats": <3|4>,  // without/with intensity
            "seed": <int>,
            // -- EXTRA PARAMETERS PER MODEL (EXAMPLES) --
            // o3d kpconv
            "sampler": "spatially_regular",
            "min_in_points": 10000,
            "max_in_points": 20000,
            "in_radius": 4.0,
            "recenter": {
                "dims": [
                    0,
                    1,
                    2
                ]
            },
            "first_subsampling_dl": 0.075,
            "conv_radius": 2.5,
            "architecture": [
                "simple",
                "resnetb",
                "resnetb_strided",
                "resnetb",
                "resnetb",
                "resnetb_strided",
                "resnetb",
                "resnetb",
                "resnetb_strided",
                "resnetb",
                "resnetb",
                "resnetb_strided",
                "resnetb",
                "nearest_upsample",
                "unary",
                "nearest_upsample",
                "unary",
                "nearest_upsample",
                "unary",
                "nearest_upsample",
                "unary"
            ],
            "num_layers": 5,
            "num_points": 45056,
            "grid_size": 0.075,
            "num_neighbors": 16,
            "sub_sampling_ratio": [
                4,
                4,
                4,
                4
            ],
            // o3d randlanet
            "sampler": "spatially_regular",
            "recenter": {
                "dims": [
                    0,
                    1
                ]
            },
            "num_points": 45056,
            "grid_size": 0.075,
            "num_neighbors": 16,
            "sub_sampling_ratio": [
                4,
                4,
                4,
                4
            ],
            // sphereformer
            "voxel_size": [
                0.05,
                0.05,
                0.05
            ],
            "voxel_max": 120000,
            "pc_range": [
                [
                    -22,
                    -17,
                    -4
                ],
                [
                    30,
                    18,
                    13
                ]
            ],
            "xyz_norm": false,
            // lsk3dnet
            "min_volume_space": [
                -120,
                -120,
                -6
            ],
            "max_volume_space": [
                120,
                120,
                11
            ]
        }
        ```
    - **ONNX**: coming soon
- Metrics:
    - Intersection over Union (IoU), Accuracy
- Computational cost:
    - Number of parameters, average inference time, model size

## Image object detection
- Datasets:
    - **[COCO](https://cocodataset.org/)**: Standard COCO format with JSON annotations and image directory structure
    - **[YOLO](https://docs.ultralytics.com/datasets/detect/)**.
- Models:
    - **PyTorch ([TorchScript](https://pytorch.org/docs/stable/jit.html) compiled format and native modules)**:
        - Input shape: `(batch, channels, height, width)`
        - Output shape: `(batch, num_detections, 6)` where each detection contains `[x1, y1, x2, y2, confidence, class_id]`
        - Output shape (torchscript-exported YOLO models): `(num_box_coords + num_classes, num_candidate_boxes)`
        - JSON configuration file format:

        ```json
        {
            "normalization": {
                "mean": [<r>, <g>, <b>],
                "std": [<r>, <g>, <b>]
            },
            "resize": {  # optional
                "width": <px>,
                "height": <px>
            },
            "confidence_threshold": <float>,
            "nms_threshold": <float>,
            "max_detections_per_image": <int>,
            "batch_size": <n>,
            "device": "<cpu|cuda|mps>",
            "evaluation_step": <int>  # for live progress updates during evaluation
            "model_format": "<coco|yolo>"
        }
        ```
- Metrics:
    - Mean Average Precision (mAP), including COCO-style mAP@[0.5:0.95:0.05]
    - Area Under the Precision-Recall Curve (AUC-PR)
    - Precision, Recall, F1-Score
    - Per-class metrics and confusion matrices with background class support
- **Evaluation Methodology (aligned with [Ultralytics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)):**
    - **mAP and Precision-Recall Curves**: Computed without confidence threshold filtering. All predicted boxes are kept to evaluate performance across the full confidence range (0 to 1). This allows proper ranking-based evaluation.
    - **Precision, Recall, and Confusion Matrix**: Computed at a specific confidence threshold:
        - If `confidence_threshold` is specified in the model config, that value is used
        - If not specified, the threshold that maximizes F1 score is automatically selected
        - The optimal threshold and corresponding F1 score are reported when auto-selection is used
    - **Confusion Matrix with Background Class**: Following Ultralytics convention, includes an implicit "background" class:
        - Unmatched predictions (FP) → counted as `predicted_class` vs `background`
        - Unmatched ground truth (FN) → counted as `background` vs `true_class`
        - This provides a complete view of model errors including missed detections
- Computational cost:
    - Number of parameters, average inference time, model size
- GUI Support:
    - Real-time inference visualization
    - Interactive dataset browsing
    - Progress tracking during evaluation