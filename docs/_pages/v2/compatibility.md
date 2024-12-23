---
layout: home
title: Compatibility
permalink: /v2/compatibility/

sidebar:
  nav: "main_v2"
---

# Image semantic segmentation
- Datasets:
    - [Rellis3D](https://www.unmannedlab.org/research/RELLIS-3D)
    - [GOOSE](https://goose-dataset.de/)
    - Custom GAIA format
- Models:
    - PyTorch ([TorchScript](https://pytorch.org/docs/stable/jit.html) format):
        - Input shape: `(batch, channels, height, width)`
        - Output shape: `(batch, classes, height, width)`
        - JSON configuration file format:

        ```json
        {
            "normalization": {
                "mean": [<r>, <g>, <b>],
                "std": [<r>, <g>, <b>]
            },
            "batch_size": 4
        }
        ```
    - Tensorflow ([SavedModel](https://www.tensorflow.org/guide/saved_mode`) format):
        - Input shape: `(batch, height, width, channels)`
        - Output shape: `(batch, height, width, classes)`
        - JSON configuration file format:

        ```json
        {
            "image_size": [<height>, <width>],
            "batch_size": 4
        }
        ```
    - ONNX: coming soon
- Metrics:
    - Intersection over Union (IoU), Accuracy

# Image semantic segmentation
- Datasets:
    - [Rellis3D](https://www.unmannedlab.org/research/RELLIS-3D)
    - [GOOSE](https://goose-dataset.de/)
    - Custom GAIA format
- Models:
    - PyTorch ([TorchScript](https://pytorch.org/docs/stable/jit.html) format). Validated models: RandLA-Net and KPConv from [Open3D-ML](https://github.com/isl-org/Open3D-ML).
        - Input shape: defined by the `input_format` tag.
        - Output shape: `(num_points)`
        - JSON configuration file format:

        ```json
        {
            "seed": 42,
            "input_format": "o3d_randlanet",
            "sampler": "spatially_regular",
            "recenter": {
                "dims": [
                    0,
                    1
                ]
            },
            "ignored_classes": [
                "void"
            ],
            "num_points": 45056,
            "grid_size": 0.06,
            "num_neighbors": 16,
            "sub_sampling_ratio": [
                4,
                4,
                4,
                4
            ]
        }
        ```
    - ONNX: coming soon
- Metrics:
    - Intersection over Union (IoU), Accuracy

# Object detection
Coming soon.