---
layout: home
title: Installation
permalink: /installation/

sidebar:
  nav: "main"
---

*PerceptionMetrics* offers two installation tracks depending on your use case:

## For Regular Users (Recommended)

In the near future, *PerceptionMetrics* will be available on PyPI. In the meantime, install directly from the repository using pip:

```bash
pip install git+https://github.com/JdeRobot/PerceptionMetrics.git
```

Or clone and install locally:
```bash
git clone https://github.com/JdeRobot/PerceptionMetrics.git
cd PerceptionMetrics
pip install .
```

### Installing Deep Learning Frameworks

Install your preferred deep learning framework:

**For PyTorch:**
```bash
# Choose one based on your needs
pip install torch==2.4.1 torchvision==0.19.1
# or
pip install torch==2.2.2 torchvision==0.17.2
```

**For TensorFlow:**
```bash
pip install tensorflow==2.17.1
```

**Note:** If you are using LiDAR, Open3D currently requires `torch==2.2*`.

We have tested the following configurations:
- CUDA Version: `12.6`
- `torch==2.4.1` and `torchvision==0.19.1`
- `torch==2.2.2` and `torchvision==0.17.2`
- `tensorflow==2.17.1`
- `tensorflow==2.16.1`

And it's done! You can check the `examples` directory for inspiration and run the provided scripts.

## For Developers

If you plan to contribute to the project or need an editable installation, use Poetry:

### 1. Install Poetry

If you haven't installed Poetry yet:
```bash
python3 -m pip install --user pipx
pipx install poetry
```

### 2. Clone and Install Dependencies

```bash
git clone https://github.com/JdeRobot/PerceptionMetrics.git
cd PerceptionMetrics
poetry install
```

### 3. Activate the Poetry Environment

```bash
poetry shell
```

You can exit the Poetry shell by running `exit`.

### 4. Install Deep Learning Frameworks

Install your deep learning framework of preference in the Poetry environment. We have tested:
- CUDA Version: `12.6`
- `torch==2.4.1` and `torchvision==0.19.1`
- `torch==2.2.2` and `torchvision==0.17.2`
- `tensorflow==2.17.1`
- `tensorflow==2.16.1`

If you are using LiDAR, Open3D currently requires `torch==2.2*`.

### Running Examples with Poetry

You can run the example scripts either by:
- Activating the environment: `poetry shell` then `python examples/<script.py>`
- Running directly: `poetry run python examples/<script.py>`

## Additional Environments

Some LiDAR segmentation models, such as SphereFormer and LSK3DNet, require a dedicated installation workflow. Refer to [additional_envs/INSTRUCTIONS.md](additional_envs/INSTRUCTIONS.md) for detailed setup instructions.