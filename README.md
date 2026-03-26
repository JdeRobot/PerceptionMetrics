<a href="https://mmg-ai.com/en/"><img src="https://jderobot.github.io/assets/images/logo.png" width="50" align="right" /></a>

# PerceptionMetrics
### _Unified evaluation for perception models_

#### Project webpage [here](https://jderobot.github.io/PerceptionMetrics)

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

![diagram](https://jderobot.github.io/PerceptionMetrics/assets/images/perceptionmetrics_diagram.png)

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

More details about the specific metrics and input/output formats required fow each framework are provided in the [Compatibility](https://jderobot.github.io/PerceptionMetrics/compatibility/) section in our webpage.

# Installation

*PerceptionMetrics* can be installed in two different ways depending on your needs:

* **Regular users**: Install the package directly from PyPI.
* **Developers**: Clone the repository and install the development environment using Poetry.

---

## Install from PyPI (Recommended for users)

The latest stable release of *PerceptionMetrics* is available on PyPI.

Install it with:

```
pip install perceptionmetrics
```

After installation, you can start using the library in your Python environment.

---

## Developer Installation

If you want to contribute to the project or modify the source code, clone the repository and install the dependencies using Poetry.

#### Clone the repository

```
git clone https://github.com/JdeRobot/PerceptionMetrics.git
cd PerceptionMetrics
```
### Using Poetry (Recommended)

Install Poetry (if not done before):
```
python3 -m pip install --user pipx
pipx install poetry
```

⚠️ Note: `pipx` should be installed **outside any virtual environment**.
If you run this command inside a `venv`, you may see:

```
ERROR: Can not perform a '--user' install. User site-packages are not visible in this virtualenv.
```


Install dependencies and activate poetry environment (you can get out of the Poetry shell by running `exit`):
```
poetry install
poetry shell
```
### Using venv
Create your virtual environment:
```
mkdir .venv
python3 -m venv .venv
```

Activate your environment and install as pip package:
```
source .venv/bin/activate
pip install -e .
```

## Common
Install your deep learning framework of preference in your environment. We have tested:
- CUDA Version: `12.6`
- `torch==2.4.1` and `torchvision==0.19.1`.
- `torch==2.2.2` and `torchvision==0.17.2`.
- `tensorflow==2.17.1`
- `tensorflow==2.16.1`
> ⚠️ **Note on PyTorch Installation (Important for Windows users)**
>
> By default, installing `torch` via `pip` may install a CUDA-enabled build (e.g., `+cu121`), which requires compatible NVIDIA drivers and CUDA setup.
>
> On systems without proper CUDA configuration, this can lead to runtime errors 
>
> To ensure a stable setup, we recommend installing the CPU-only version of PyTorch unless GPU support is explicitly required:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```
>
> GPU users can install CUDA-specific builds manually based on their system configuration.

If you are using LiDAR, Open3D currently requires `torch==2.2*`.

And it's done! You can check the `examples` directory for inspiration and run some of the scripts provided either by activating the created environment using `poetry shell` or directly running `poetry run python examples/<some_python_script.py>`.

### Additional environments
Some LiDAR segmentation models, such as SphereFormer and LSK3DNet, require a dedicated installation workflow. Refer to [additional_envs/INSTRUCTIONS.md](additional_envs/INSTRUCTIONS.md) for detailed setup instructions.
# Usage
PerceptionMetrics can be used in three ways: through the **interactive GUI** (detection only), as a **Python library**, or via the **command-line interface** (segmentation and detection).

## Interactive GUI
The easiest way to get started with PerceptionMetrics is through the GUI (detection tasks only):

```bash
# From the project root directory
streamlit run app.py
```

The GUI provides:
- **Dataset Viewer**: Browse and visualize your datasets
- **Inference**: Run real-time inference on images
- **Evaluator**: Perform comprehensive model evaluation

For detailed GUI documentation, see our [GUI guide](https://jderobot.github.io/PerceptionMetrics/gui).

## Library

🧑‍🏫️ [Image Segmentation Tutorial](https://github.com/JdeRobot/PerceptionMetrics/blob/master/examples/tutorial_image_segmentation.ipynb)

🧑‍🏫️ [Image Detection Tutorial](https://github.com/JdeRobot/PerceptionMetrics/blob/master/examples/tutorial_image_detection.ipynb)

🧑‍🏫️ [Image Detection Tutorial (YOLO)](https://github.com/JdeRobot/PerceptionMetrics/blob/master/examples/tutorial_image_detection_yolo.ipynb)

You can check the `examples` directory for further inspiration. If you are using *poetry*, you can run the scripts provided either by activating the created environment using `poetry shell` or directly running `poetry run python examples/<some_python_script.py>`.

## Command-line interface
PerceptionMetrics provides a CLI with two commands, `pm_evaluate` and `pm_batch`. Thanks to the configuration in the `pyproject.toml` file, we can simply run `poetry install` from the root directory and use them without explicitly invoking the Python files. More details are provided in [PerceptionMetrics website](https://jderobot.github.io/PerceptionMetrics/usage/#command-line-interface).

### Example Usage
**Segmentation:**
```bash
pm_evaluate segmentation image --model_format torch --model /path/to/model.pt --model_ontology /path/to/ontology.json --model_cfg /path/to/cfg.json --dataset_format rellis3d --dataset_dir /path/to/dataset --dataset_ontology /path/to/ontology.json --out_fname /path/to/results.csv
```

**Detection:**
```bash
pm_evaluate detection image --model_format torch --model /path/to/model.pt --model_ontology /path/to/ontology.json --model_cfg /path/to/cfg.json --dataset_format coco --dataset_dir /path/to/coco/dataset --out_fname /path/to/results.csv
```

<h1 id="DetectionMetrics">DetectionMetrics</h1>

Our previous release, ***DetectionMetrics***, introduced a versatile suite focused on object detection, supporting cross-framework evaluation and analysis. [Cite our work](#cite) if you use it in your research!

<table style='font-size:100%'>
  <tr>
    <th>&#128187; <a href="https://github.com/JdeRobot/PerceptionMetrics/releases/tag/v1.0.0">Code</a></th>
    <th>&#128214; <a href="https://jderobot.github.io/PerceptionMetrics/DetectionMetrics">Docs</a></th>
    <th>&#128011; <a href="https://hub.docker.com/r/jderobot/detection-metrics">Docker</a></th>
    <th>&#128240; <a href="https://www.mdpi.com/1424-8220/22/12/4575">Paper</a></th>
  </tr>
</table>

<h1 id="cite">Cite our work</h1>

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

# How to Contribute
_To make your first contribution, follow this [Guide](https://github.com/JdeRobot/PerceptionMetrics/blob/master/CONTRIBUTING.md)._

# Acknowledgements
LiDAR segmentation support is built upon open-source work from [Open3D-ML](https://github.com/isl-org/Open3D-ML), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [SphereFormer](https://github.com/dvlab-research/SphereFormer), and [LSK3DNet](https://github.com/FengZicai/LSK3DNet).
