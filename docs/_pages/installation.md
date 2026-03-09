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

### Install Poetry

Poetry is used to manage dependencies and development environments.

First install `pipx` (recommended for installing CLI tools):

```
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

Then install Poetry:

```
pipx install poetry
```

⚠️ Note: `pipx` should be installed **outside any virtual environment**.
If you run this command inside a `venv`, you may see:

```
ERROR: Can not perform a '--user' install. User site-packages are not visible in this virtualenv.
```

### Install dependencies

Poetry automatically creates and manages a virtual environment for the project.

Install dependencies with:
```
poetry install
```

Activate the environment:

```
poetry shell
```

You can exit the Poetry environment anytime by running:

```
exit
```

---

## Deep Learning Framework Setup

Install the deep learning framework of your choice inside your environment.

The following configurations have been tested:

* CUDA Version: `12.6`
* `torch==2.4.1` and `torchvision==0.19.1`
* `torch==2.2.2` and `torchvision==0.17.2`
* `tensorflow==2.17.1`
* `tensorflow==2.16.1`

If you are working with LiDAR models, note that **Open3D currently requires**:

```
torch==2.2*
```

---

## Running Examples

After installation, you can explore the examples provided in the repository.

If using Poetry:

```
poetry run python examples/<some_script.py>
```

or activate the environment:

```
poetry shell
python examples/<some_script.py>
```

---

### Additional Environments

Some LiDAR segmentation models, such as **SphereFormer** and **LSK3DNet**, require additional installation steps.

Refer to [additional_envs/INSTRUCTIONS.md](additional_envs/INSTRUCTIONS.md) for detailed setup instructions.
