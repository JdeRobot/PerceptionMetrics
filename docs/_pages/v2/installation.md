---
layout: home
title: Installation
permalink: /v2/installation/

sidebar:
  nav: "main_v2"
---
In the near future, *DetectionMetrics* is planned to be deployed in PyPI. In the meantime, you can clone our repo and build the package locally using either *venv* or *Poetry*.

```
git clone git@github.com:JdeRobot/DetectionMetrics.git && cd DetectionMetrics
```

## Using venv
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

## Using Poetry

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

## Common
Install your deep learning framework of preference in your environment. We have tested:
- CUDA Version: `12.6`
- `torch==2.4.1`
- `torchvision==0.19.1`
- `tensorflow[and-cuda]==2.17.1`

If you are using LiDAR, Open3D currently requires `torch==2.0*`.

And it's done! You can check the `examples` directory for inspiration and run some of the scripts provided either by activating the created environment using `poetry shell` or directly running `poetry run python examples/<some_python_script.py>`.