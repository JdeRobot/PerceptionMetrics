---
layout: splash
title: PerceptionMetrics
permalink: /
header:
  overlay_color: "#5e616c"
  overlay_image: /assets/images/cover/test_header_shear_3.png
  actions:
   - label: "<i class='fas fa-download'></i> Learn more and contribute"
     url: "https://github.com/JdeRobot/PerceptionMetrics"
excerpt:
  Unified evaluation for perception models
---
>&#9888;&#65039; PerceptionMetrics was previously known as DetectionMetrics. The original website referenced in our *Sensors* paper is still available [here](https://jderobot.github.io/PerceptionMetrics/DetectionMetrics)

*PerceptionMetrics* is a toolkit designed to unify and streamline the evaluation of object detection and segmentation models across different sensor modalities, frameworks, and datasets. It offers multiple interfaces including a GUI for interactive analysis, a CLI for batch evaluation, and a Python library for seamless integration into your codebase. The toolkit provides consistent abstractions for models, datasets, and metrics, enabling fair, reproducible comparisons across heterogeneous perception systems.

<table style='font-size:100%; margin: auto;'>
  <tr>
    <th>&#128187; <a href="https://github.com/JdeRobot/PerceptionMetrics">Code</a></th>
    <th>&#128295; <a href="https://jderobot.github.io/PerceptionMetrics/installation">Installation</a></th>
    <th>&#129513; <a href="https://jderobot.github.io/PerceptionMetrics/compatibility">Compatibility</a></th>
    <th>&#128214; <a href="https://jderobot.github.io/PerceptionMetrics/py_docs/build/html/index.html">Docs</a></th>
    <th>&#128187; <a href="https://jderobot.github.io/PerceptionMetrics/gui">GUI</a></th>
  </tr>
</table>

![diagram](../assets/images/perceptionmetrics_diagram.png)

# What's supported in PerceptionMetrics

<table><thead>
  <tr>
    <th>Task</th>
    <th>Modality</th>
    <th>Datasets</th>
    <th>Framework</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="2">Segmentation</td>
    <td>Image</td>
    <td>RELLIS-3D, GOOSE, RUGD, WildScenes, custom GAIA format</td>
    <td>PyTorch, Tensorflow</td>
  </tr>
  <tr>
    <td>LiDAR</td>
    <td>RELLIS-3D, GOOSE, WildScenes, custom GAIA format</td>
    <td>PyTorch (tested with <a href="https://github.com/isl-org/Open3D-ML">Open3D-ML</a>, <a href="https://github.com/open-mmlab/mmdetection3d">mmdetection3d</a>, <a href="https://github.com/dvlab-research/SphereFormer">SphereFormer</a>, and <a href="https://github.com/FengZicai/LSK3DNet">LSK3DNet</a> models)</td>  </tr>
  <tr>
    <td>Object detection</td>
    <td>Image</td>
    <td>COCO, YOLO</td>
    <td>PyTorch (tested with torchvision and torchscript-exported YOLO models)</td>
  </tr>
</tbody>
</table>

More details about the specific metrics and input/output formats required fow each framework are provided in the [Compatibility](https://jderobot.github.io/PerceptionMetrics/compatibility/) section

# DetectionMetrics

Our previous release, ***DetectionMetrics***, introduced a versatile suite focused on object detection, supporting cross-framework evaluation and analysis. [Cite our work](#cite) if you use it in your research!

<table style='font-size:100%'>
  <tr>
    <th>&#128187; <a href="https://github.com/JdeRobot/PerceptionMetrics/releases/tag/v1.0.0">Code</a></th>
    <th>&#128214; <a href="https://jderobot.github.io/PerceptionMetrics/DetectionMetrics">Docs</a></th>
    <th>&#128011; <a href="https://hub.docker.com/r/jderobot/detection-metrics">Docker</a></th>
    <th>&#128240; <a href="https://www.mdpi.com/1424-8220/22/12/4575">Paper</a></th>
  </tr>
</table>

# Cite our work
{: #cite}

```
@article{PaniegoOSAssessment2022,
  author = {Paniego, Sergio and Sharma, Vinay and Cañas, José María},
  title = {Open Source Assessment of Deep Learning Visual Object Detection},
  journal = {Sensors},
  volume = {22},
  year = {2022},
  number = {12},
  article-number = {4575},
  url = {https://www.mdpi.com/1424-8220/22/12/4575},
  pubmedid = {35746357},
  issn = {1424-8220},
  doi = {10.3390/s22124575},
}
```