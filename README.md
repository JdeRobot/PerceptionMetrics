<a href="https://mmg-ai.com/en/"><img src="https://jderobot.github.io/assets/images/logo.png" width="50" align="right" /></a>

# DetectionMetrics

#### Project webpage [here](https://jderobot.github.io/DetectionMetrics)

>&#9888;&#65039; ***DetectionMetrics v1*** website referenced in our *Sensors* paper is still available [here](https://jderobot.github.io/DetectionMetrics/v1)

*DetectionMetrics* is a family of toolkits designed to unify and streamline the evaluation of perception models across different frameworks and datasets. Looking for our published ***DetectionMetrics v1***? Check out all the [relevant links](#v1) below.

Now, we're excited to introduce ***DetectionMetrics v2***! While retaining the flexibility of our previous release, *DetectionMetrics* has been redesigned with an expanded focus on image segmentation, with plans to extend support to object detection and LiDAR applications. As we move forward, *v2* will be the actively maintained version, featuring continued updates and enhancements to keep pace with evolving AI and computer vision technologies.

<table style='font-size:100%'>
  <tr>
    <th>&#128187; <a href="https://github.com/JdeRobot/DetectionMetrics">Code</a></th>
    <th>&#128295; <a href="https://jderobot.github.io/DetectionMetrics/v2/installation">Installation</a></th>
    <th>&#129513; <a href="https://jderobot.github.io/DetectionMetrics/v2/compatibility">Compatibility</a></th>
    <th>&#128214; <a href="https://jderobot.github.io/DetectionMetrics/py_docs/_build/html/index.html">Docs</a></th>
  </tr>
</table>

![diagram](docs/assets/images/detectionmetricsv2_diagram.png)

# What's supported in DetectionMetrics

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
    <td>Rellis3D, GOOSE, custom GAIA format</td>
    <td>PyTorch, Tensorflow</td>
  </tr>
  <tr>
    <td>LiDAR</td>
    <td>Coming soon</td>
    <td>Coming soon</td>
  </tr>
  <tr>
    <td>Object detection</td>
    <td>Image</td>
    <td>Check <a href="https://jderobot.github.io/DetectionMetrics/v1"><i>DetectionMetrics v1</i></a></td>
    <td>Check <a href="https://jderobot.github.io/DetectionMetrics/v1"><i>DetectionMetrics v1</i></a></td>
  </tr>
</tbody>
</table>

More details about the specific metrics and input/output formats required fow each framework are provided in the [Compatibility](/v2/compatibility) section in our webpage.


# Installation
In the near future, *DetectionMetrics* is planned to be deployed in PyPI. In the meantime, you can clone our repo and build the package locally using either *venv* or *Poetry*.

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

### Using Poetry

Install Poetry (if not done before):
```
python3 -m pip install --user pipx
pipx install poetry
```

Install dependencies and activate poetry environment (you can get out of the Poetry shell by running `exit`):
```
poetry install
poetry shell
```

### Common
Install your deep learning framework of preference in your environment. We have tested:
- CUDA Version: `12.6`
- `torch==2.4.1`
- `torchvision==0.19.1`
- `tensorflow[and-cuda]==2.17.1`

And it's done! You can check the `examples` directory for inspiration and run some of the scripts provided either by activating the created environment using `poetry shell` or directly running `poetry run python examples/<some_python_script.py>`.

<h1 id="v1">DetectionMetrics v1</h1>

Our previous release, ***DetectionMetrics v1***, introduced a versatile suite focused on object detection, supporting cross-framework evaluation and analysis. [Cite our work](#cite) if you use it in your research!

<table style='font-size:100%'>
  <tr>
    <th>&#128187; <a href="https://github.com/JdeRobot/DetectionMetrics/releases/tag/v1.0.0">Code</a></th>
    <th>&#128214; <a href="https://jderobot.github.io/DetectionMetrics/v1">Docs</a></th>
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
