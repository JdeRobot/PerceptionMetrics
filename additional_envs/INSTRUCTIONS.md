# Additional environments

Some LiDAR segmentation backends require a dedicated Python version and separate virtual environment.
These setups are not compatible with the default installation workflow.

---

## MMDetection3D

### Python version
- **Python 3.10** (recommended)

### Create and activate a virtual environment
```bash
python3.10 -m venv .venv-mmdet3d
source .venv-mmdet3d/bin/activate
python -m pip install -U pip setuptools wheel
```

### Install dependencies (CUDA 11.7)
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
mim install "mmdet>=3.0.0"
mim install "mmdet3d>=1.1.0"
```

### Install TorchSparse

#### Option A (with sudo)
```bash
sudo apt update
sudo apt install -y gcc-11 g++-11 nvidia-cuda-toolkit python3.10-dev
sudo apt install -y libsparsehash-dev

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export FORCE_CUDA=1
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

#### Option B (no sudo / Option A fails)

##### 1) Install CUDA 11.7 locally
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
chmod +x cuda_11.7.0_515.43.04_linux.run

mkdir -p "$HOME/cuda-11.7"
./cuda_11.7.0_515.43.04_linux.run \
  --toolkit --override \
  --installpath="$HOME/cuda-11.7"

export CUDA_HOME="$HOME/cuda-11.7"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

##### 2) Install Google's SparseHash locally
```bash
PREFIX=$HOME/local
mkdir -p "$PREFIX" && cd /tmp
wget -q https://github.com/sparsehash/sparsehash/archive/refs/tags/sparsehash-2.0.4.tar.gz
tar xzf sparsehash-2.0.4.tar.gz
cd sparsehash-sparsehash-2.0.4
./configure --prefix="$PREFIX"
make -j"$(nproc)" && make install   # headers land in $PREFIX/include/google/
export CPLUS_INCLUDE_PATH="$PREFIX/include:$CPLUS_INCLUDE_PATH"
```

##### 3) Install TorchSparse
```bash
export FORCE_CUDA=1
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

---

## SphereFormer

### Python version
- **Python 3.7** (required)

### Create and activate a virtual environment
```bash
python3.7 -m venv .venv-sphereformer
source .venv-sphereformer/bin/activate
python -m pip install -U pip setuptools wheel
```

### Install dependencies
```bash
pip install typing-extensions==4.7.1

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install torch_scatter==2.0.9
pip install torch_geometric==1.7.2
pip install spconv-cu114==2.1.25
pip install torch_sparse==0.6.12 cumm-cu114==0.2.8 torch_cluster==1.5.9

pip install safetensors==0.3.3
pip install tensorboard timm termcolor tensorboardX
```

### Clone SphereFormer and build its SparseTransformer
```bash
mkdir -p third_party && cd third_party
git clone https://github.com/dvlab-research/SphereFormer.git
cd SphereFormer/third_party/SparseTransformer
python setup.py install
```

### Switch to the SphereFormer-specific pyproject.toml and install PerceptionMetrics
Run the following from the repository root:
```bash
cd ../../..
mv pyproject.toml pyproject-core.toml
cp additional_envs/pyproject-sphereformer.toml pyproject.toml

pip install -e .
```

### Add SphereFormer to PYTHONPATH
Run the following from the repository root:
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/third_party/SphereFormer"
```

---

## LSK3DNet

### Python version
- **Python 3.9** (required)

Ensure `python3.9-dev` and `python3.9-distutils` are available.

### Create and activate a virtual environment
```bash
python3.9 -m venv .venv-lsk3dnet
source .venv-lsk3dnet/bin/activate
python -m pip install -U pip setuptools wheel
```

### Install dependencies (CUDA 11.3)
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

pip install numpy==1.23.5
pip install -r additional_envs/requirements-lsk3dnet.txt

pip install SharedArray==3.2.4
pip install pybind11
```

### Build LSK3DNet c_utils
```bash
mkdir -p third_party && cd third_party
git clone https://github.com/FengZicai/LSK3DNet.git

cd LSK3DNet/c_utils
mkdir -p build && cd build

cmake -DPYTHON_EXECUTABLE="$(which python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  ..

make
```

### Switch to the LSK3DNet-specific pyproject.toml and install PerceptionMetrics
Run the following from the repository root:
```bash
cd ../../../..

mv pyproject.toml pyproject-core.toml
cp additional_envs/pyproject-lsk3dnet.toml pyproject.toml

pip install -e .
```

### Add LSK3DNet to PYTHONPATH
Run the following from the repository root:
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/third_party/LSK3DNet:$(pwd)/third_party/LSK3DNet/c_utils/build"
```

---

## Restore the core repository configuration

If you switched `pyproject.toml` for a backend-specific installation, restore the default setup from the repository root:

```bash
mv pyproject-core.toml pyproject.toml
```
