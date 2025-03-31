---
layout: home
title: Compatibility
permalink: /v2/compatibility/

sidebar:
  nav: "main_v2"
---

## Image semantic segmentation
- Datasets:
    - **[Rellis3D](https://www.unmannedlab.org/research/RELLIS-3D)**
    - **[GOOSE](https://goose-dataset.de/)**
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
    - **Custom GAIA format**: *Parquet* file containing samples and labels relative paths and a JSON file with the dataset ontology.
    - **Generic**: simply assumes a different directory per split, different suffixes for samples and labels, and a JSON file containing the dataset ontology.
- Models:
    - **PyTorch ([TorchScript](https://pytorch.org/docs/stable/jit.html) compiled format and native modules)**. As of now, we have tested RandLA-Net and KPConv from [Open3D-ML](https://github.com/isl-org/Open3D-ML).
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
    - **ONNX**: coming soon
- Metrics:
    - Intersection over Union (IoU), Accuracy
- Computational cost:
    - Number of parameters, average inference time, model size

## Object detection
Coming soon.