---
layout: home
title: Datasets
permalink: /datasets/

sidebar:
  nav: "main"
---

This page summarizes the dataset adapters available in PerceptionMetrics and the inputs each adapter expects. Dataset adapters normalize different dataset layouts into the common `PerceptionDataset` abstractions used by model evaluation, prediction evaluation, the GUI, and the Python API.

## Support Matrix

| Dataset adapter | Task | Modality | CLI format | Status |
| --- | --- | --- | --- | --- |
| GAIA | Segmentation | Image, LiDAR | `gaia` | CLI and library |
| Generic | Segmentation | Image, LiDAR | `generic` | CLI and library |
| [GOOSE](https://goose-dataset.de/) | Segmentation | Image, LiDAR | `goose` | CLI and library |
| [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) | Segmentation | Image, LiDAR | `rellis3d` | CLI and library |
| [RUGD](http://rugd.vision/) | Segmentation | Image | `rugd` | CLI and library |
| [WildScenes](https://github.com/csiro-robotics/WildScenes) | Segmentation | Image, LiDAR | `wildscenes` | CLI and library |
| [Cityscapes](https://www.cityscapes-dataset.com/) | Segmentation | Image | `cityscapes` | CLI and library |
| [SemanticKITTI](https://semantic-kitti.org/) | Segmentation | LiDAR | `semantickitti_lidar_segmentation` | CLI and library |
| [COCO](https://cocodataset.org/#home) | Object detection | Image | `coco` | CLI and library |
| [YOLO](https://docs.ultralytics.com/datasets/detect/) | Object detection | Image | `yolo` | CLI and library |
| [nuImages](https://www.nuscenes.org/nuimages) | Segmentation, object detection | Image | `nuimages` | CLI and library |

The CLI constructs datasets through `perceptionmetrics.cli.get_dataset`, while the same adapters can also be used directly from Python.

## Common Concepts

All segmentation datasets expose samples with a data file, label file, and split. Image segmentation datasets use image masks as labels; LiDAR segmentation datasets use point labels. Object detection datasets expose image files and annotations that can be converted into bounding boxes and class indices.

Most segmentation adapters need an ontology. The common ontology shape is:

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

Some adapters build the ontology from the official dataset metadata. For others, you need to provide an ontology file. The ontology file can be in JSON or YAML format, and the exact expected shape depends on the adapter. 



## GAIA

API: [`GaiaImageSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.gaia.GaiaImageSegmentationDataset" | relative_url }}), [`GaiaLiDARSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.gaia.GaiaLiDARSegmentationDataset" | relative_url }}).

GAIA is a custom PerceptionMetrics format backed by a Parquet file. The Parquet file is loaded with Pandas, and paths are resolved relative to the directory containing that Parquet file.

Expected inputs:

- `--dataset_fname /path/to/dataset.parquet`
- An ontology file in the same directory, named `ontology.json` by default
- Optionally, a Parquet attribute named `ontology_fname` pointing to a different ontology filename

Typical columns:

| Modality | Required columns |
| --- | --- |
| Image segmentation | `image`, `label`, `split` |
| LiDAR segmentation | `points`, `label`, `split` |

## Generic

API: [`GenericImageSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.generic.GenericImageSegmentationDataset" | relative_url }}), [`GenericLiDARSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.generic.GenericLiDARSegmentationDataset" | relative_url }}).

Generic datasets are useful when files can be paired by matching wildcard captures in input and label patterns. They support image and LiDAR segmentation.

Expected inputs:

- At least one of `--train_dataset_dir`, `--val_dataset_dir`, or `--test_dataset_dir`
- `--data_suffix`
- `--label_suffix`
- `--dataset_ontology`

The data and label suffixes must contain the same number of `*` wildcards. For example:

```shell
--data_suffix "*_image.png" --label_suffix "*_label.png"
```

The ontology may be either a list of class names:

```json
["background", "road", "vegetation"]
```

or a dictionary:

```json
{
  "background": {"idx": 0, "rgb": [0, 0, 0]},
  "road": {"idx": 1, "rgb": [128, 64, 128]}
}
```

### Image file-pairing example

The following paths are relative to a custom dataset root:

| File | Purpose |
| --- | --- |
| `ontology.json` | JSON class list or dictionary, as above |
| `val/frame001_image.png` | Input image |
| `val/frame001_label.png` | Single-channel mask containing class IDs |

```python
from perceptionmetrics.datasets.generic import GenericImageSegmentationDataset

dataset = GenericImageSegmentationDataset(
    data_suffix="*_image.png",
    label_suffix="*_label.png",
    ontology_fname="/path/to/custom/ontology.json",
    val_dataset_dir="/path/to/custom/val",
)
```

The wildcard captures `frame001` and pairs the image with
`frame001_label.png`. In the CLI, `ontology_fname` corresponds to
`--dataset_ontology`; the other keyword names correspond to the same-named
CLI options. A missing matching label raises an error. Use distinct captured
sample names across splits because samples are indexed by those names.

For `GenericImageSegmentationDataset`, masks should store class indices as
8-bit grayscale values, not visualization colors. With the dictionary above,
a pixel value of `1` means `road`; the `rgb` entry is for visualization.
The Generic adapter reads its ontology as JSON, not YAML.

This image-mask example does not describe LiDAR labels. The Generic LiDAR
adapter inherits the base SemanticKITTI-style binary label reader; see
[`LiDARSegmentationDataset.read_label`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.segmentation.LiDARSegmentationDataset.read_label" | relative_url }}).

## GOOSE

API: [`GOOSEImageSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.goose.GOOSEImageSegmentationDataset" | relative_url }}), [`GOOSELiDARSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.goose.GOOSELiDARSegmentationDataset" | relative_url }}).

GOOSE supports image and LiDAR semantic segmentation. The adapter expects the official split directory layout and reads `goose_label_mapping.csv` from the first provided split root.

Expected inputs:

- One or more of `--train_dataset_dir`, `--val_dataset_dir`, `--test_dataset_dir`
- Each split root should contain `goose_label_mapping.csv`
- Image data under `images/<split>/*/*_windshield_vis.png`
- Image labels under `labels/<split>/<scene>/*_labelids.png`
- LiDAR data under `lidar/<split>/*/*_vls128.bin` for GOOSE or `*_pcl.bin` for GOOSE Ex
- LiDAR labels under `labels/<split>/<scene>/*_goose.label`

## RELLIS-3D

API: [`Rellis3DImageSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.rellis3d.Rellis3DImageSegmentationDataset" | relative_url }}), [`Rellis3DLiDARSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.rellis3d.Rellis3DLiDARSegmentationDataset" | relative_url }}).

RELLIS-3D supports image and LiDAR semantic segmentation. The adapter uses official `.lst` split files and a YAML ontology file.

Expected inputs:

- `--dataset_dir`: directory containing the extracted data and labels
- `--split_dir`: directory containing split files
- `--dataset_ontology`: RELLIS-3D ontology YAML

Image split files:

- `train.lst`
- `val.lst`
- `test.lst`

LiDAR split files:

- `pt_train.lst`
- `pt_val.lst`
- `pt_test.lst`

Each row in a split file is expected to contain the relative data path and relative label path separated by a space.

## RUGD

API: [`RUGDImageSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.rugd.RUGDImageSegmentationDataset" | relative_url }}).

RUGD supports image semantic segmentation. Labels are RGB masks, so the adapter initializes the base image segmentation dataset with RGB-label handling enabled.

Expected inputs:

- `--images_dir`
- `--labels_dir`
- `--dataset_ontology`, usually `RUGD_annotation-colormap.txt`

The default train, validation, and test split assignment is built into the adapter using the sequence names from the RUGD paper.

## WildScenes

API: [`WildscenesImageSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.wildscenes.WildscenesImageSegmentationDataset" | relative_url }}), [`WildscenesLiDARSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.wildscenes.WildscenesLiDARSegmentationDataset" | relative_url }}).

WildScenes supports image and LiDAR semantic segmentation through the CLI and Python API. The adapters expect official CSV split files and use ontology definitions embedded in the adapter source.

Expected inputs:

- `dataset_dir`: root of the WildScenes data
- `split_dir`: directory containing `train.csv`, `val.csv`, and `test.csv`

Use the 2D split files for `WildscenesImageSegmentationDataset` and the 3D split files for `WildscenesLiDARSegmentationDataset`.

## Cityscapes

API: [`CityscapesImageSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.cityscapes.CityscapesImageSegmentationDataset" | relative_url }}).

Cityscapes supports image semantic segmentation through the CLI and Python API.

Expected inputs:

- One or more of `train_dataset_root`, `val_dataset_root`, `test_dataset_root`
- Images under `leftImg8bit_trainvaltest/leftImg8bit/<split>/<city>/`
- Labels under `gtFine/<split>/<city>/`
- Default image suffix: `_leftImg8bit.png`
- Default label suffix: `_gtFine_labelIds.png`

The adapter can build either Cityscapes label-id ontologies or train-id ontologies. When using train IDs, provide train-id labels with `label_suffix="_gtFine_labelTrainIds.png"`.

## SemanticKITTI

API: [`SemanticKITTILiDARSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.semantickitti.SemanticKITTILiDARSegmentationDataset" | relative_url }}).

SemanticKITTI supports LiDAR semantic segmentation through the CLI and Python API.

Expected inputs:

- `dataset_dir`: directory where SemanticKITTI has been extracted
- `config_fname`: SemanticKITTI YAML config containing labels, colors, learning maps, and splits
- `split`, optionally restricting the loaded samples to one split

The adapter reads point clouds from `velodyne` folders and labels from `labels` folders. It can build raw label-ID ontologies and train-ID ontology translations from the SemanticKITTI YAML config.

## COCO

API: [`CocoDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.coco.CocoDataset" | relative_url }}).

COCO supports image object detection and is available from the CLI and Python API.

Expected layout:

```text
dataset_root/
  images/
    train2017/
    val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

Expected CLI inputs:

- `--dataset_format coco`
- `--dataset_dir /path/to/dataset_root`
- `--split train` or `--split val`

The CLI currently supports one COCO split at a time. The adapter looks for an image directory matching the split name, such as `val2017`, and an annotation file matching `instances_<split>*.json`.

### Minimal annotation example

For an image `images/val2017/frame001.jpg` with width 640 and height 480,
`annotations/instances_val2017.json` can contain:

```json
{
  "images": [{"id": 1, "file_name": "frame001.jpg", "width": 640, "height": 480}],
  "categories": [{"id": 1, "name": "person"}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [100, 50, 80, 120], "area": 9600, "iscrowd": 0}
  ]
}
```

`file_name` is relative to the image directory. `image_id` connects an
annotation to an image, while `category_id` refers to a category entry.
COCO boxes use absolute pixel `[x, y, width, height]` coordinates.
`read_annotation("1")` returns `([[100, 50, 180, 170]], [1])`: boxes are
converted to XYXY, and category IDs are preserved rather than renumbered.

```python
from perceptionmetrics.datasets.coco import CocoDataset

dataset = CocoDataset(
    annotation_file="/path/to/dataset_root/annotations/instances_val2017.json",
    image_dir="/path/to/dataset_root/images/val2017",
    split="val",
)
```

The Python constructor takes explicit paths; the CLI discovers those paths
from `--dataset_dir` and `--split` using
[`find_img_dir_and_ann_file`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.coco.find_img_dir_and_ann_file" | relative_url }}).
The ontology is built from `categories`, so no separate dataset ontology file
is required. See the existing [COCO detection tutorial](https://github.com/JdeRobot/PerceptionMetrics/blob/master/examples/tutorial_image_detection.ipynb)
for model evaluation.

## YOLO

API: [`YOLODataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.yolo.YOLODataset" | relative_url }}).

YOLO supports image object detection through the CLI and Python API. The adapter reads an Ultralytics-style dataset YAML file.

Expected YAML fields:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
names:
  0: person
  1: vehicle
```

The adapter expects labels in matching `labels/<split>` directories and converts YOLO normalized center-width-height boxes into absolute `[x1, y1, x2, y2]` boxes.

### Matching images and annotations

For the YAML above, a validation sample uses these paths:

| File | Purpose |
| --- | --- |
| `/path/to/dataset/images/val/frame001.jpg` | Image, for example 640 × 480 pixels |
| `/path/to/dataset/labels/val/frame001.txt` | One annotation row per object |

Example annotation row:

```text
0 0.5 0.5 0.25 0.5
```

The fields are `class_id center_x center_y width height`. Coordinates and
sizes are normalized by image width/height. For this 640 × 480 image, the row
represents class `0` (`person`) and converts to `[240, 120, 400, 360]` in
absolute XYXY coordinates.

```python
from perceptionmetrics.datasets.yolo import YOLODataset

dataset = YOLODataset(
    dataset_fname="/path/to/data.yaml",
    dataset_dir="/path/to/dataset",
    im_ext="jpg",
)
```

`dataset_fname` corresponds to `--dataset_fname` and `dataset_dir` to
`--dataset_dir`. Pass `dataset_dir=None` explicitly to use the YAML `path`;
an absolute `path` avoids dependence on the current working directory.
Relative split paths are resolved against that dataset root. The default
image extension is `jpg`; set `im_ext="png"` in Python for PNG images.

The adapter includes a sample only when both image and label files exist.
Use an empty label file for an image with no annotated objects. Keep image
stems unique across splits because they are used as sample keys.
The YAML `names` field supplies the ontology. See the existing
[YOLO detection tutorial](https://github.com/JdeRobot/PerceptionMetrics/blob/master/examples/tutorial_image_detection_yolo.ipynb)
for model evaluation.

## nuImages

API: [`NuImagesDetectionDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.nuimages.NuImagesDetectionDataset" | relative_url }}), [`NuImagesSegmentationDataset`]({{ "/py_docs/build/html/api/perceptionmetrics.datasets.html#perceptionmetrics.datasets.nuimages.NuImagesSegmentationDataset" | relative_url }}).

nuImages supports image object detection and image semantic segmentation through the CLI and Python API.

Expected inputs:

- `dataset_dir`: nuImages root directory
- `version`, defaulting to `v1.0-mini`
- `split`, defaulting to `train`
