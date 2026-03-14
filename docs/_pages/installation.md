---
layout: home
title: Installation
permalink: /installation/

sidebar:
  nav: "main"
---

## Installation

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

## Developer Installation (Using Poetry)

If you want to contribute to the project or modify the source code, clone the repository and install the dependencies using Poetry.

### Clone the repository

```
git clone https://github.com/JdeRobot/PerceptionMetrics.git
cd PerceptionMetrics
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
### Using Poetry

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

## Common
Install your deep learning framework of preference in your environment. We have tested:
- CUDA Version: `12.6`
- `torch==2.4.1` and `torchvision==0.19.1`.
- `torch==2.2.2` and `torchvision==0.17.2`.
- `tensorflow==2.17.1`
- `tensorflow==2.16.1`

If you are using LiDAR, Open3D currently requires `torch==2.2*`.

And it's done! You can check the `examples` directory for inspiration and run some of the scripts provided either by activating the created environment using `poetry shell` or directly running `poetry run python examples/<some_python_script.py>`.

### Additional environments
Some LiDAR segmentation models, such as SphereFormer and LSK3DNet, require a dedicated installation workflow. Refer to [additional_envs/INSTRUCTIONS.md](additional_envs/INSTRUCTIONS.md) for detailed setup instructions.
