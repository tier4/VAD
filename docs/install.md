# Installation Guide

This guide provides step-by-step instructions for installing VAD with either CUDA 11.8 or CUDA 12.8 support.

## Prerequisites

- **Miniforge/Mamba**: We use mamba for environment management. Install from [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge)
- **CUDA-capable GPU**: Required for training and inference
- **GCC 9.4**: Strongly recommended for compiling CUDA extensions

## Choose Your Installation Path

Select based on your CUDA version requirements:
- **CUDA 11.8**: Uses Python 3.8, stable PyTorch release
- **CUDA 12.8**: Uses Python 3.10, PyTorch nightly builds

---

## Option 1: CUDA 11.8 Installation (Python 3.8)

### Step 1: Create Virtual Environment
```bash
mamba create -n vad python=3.8 -y
mamba activate vad
```

### Step 2: Install CUDA Toolkit
```bash
mamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

### Step 3: Install PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Compilers
```bash
# GCC 9.4 is required for CUDA 11.8 compatibility
mamba install -c conda-forge c-compiler cxx-compiler gcc_linux-64=9.4 gxx_linux-64=9.4
```

### Step 5: Install Package
```bash
pip install ninja packaging
pip install -e ".[all]" --no-build-isolation
```

**Note**: Python 3.8 will automatically install scipy==1.7.3 (latest compatible version)

---

## Option 2: CUDA 12.8 Installation (Python 3.10)

### Step 1: Create Virtual Environment
```bash
mamba create -n vad python=3.10 -y
mamba activate vad
```

### Step 2: Install CUDA Toolkit
```bash
mamba install -c "nvidia/label/cuda-12.8.0" cuda-toolkit
```

### Step 3: Install PyTorch (Nightly)
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Step 4: Install Compilers
```bash
# GCC 9.4 is recommended for consistency
mamba install -c conda-forge c-compiler cxx-compiler gcc_linux-64=9.4 gxx_linux-64=9.4
```

### Step 5: Install Package
```bash
pip install ninja packaging
pip install -e ".[all]" --no-build-isolation
```

**Note**: Python 3.10 will automatically install scipy==1.13.1 (latest version)

---

## Common Final Steps (Both CUDA Versions)

### Download Pretrained Weights

1. Create checkpoint directory:
   ```bash
   mkdir ckpts
   ```

2. Download required pretrained models:
   - **ResNet50**: `resnet50-19c8e357.pth`
     - [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/resnet50-19c8e357.pth)
     - [Baidu Cloud](https://pan.baidu.com/s/1LlSrbYvghnv3lOlX1uLU5g?pwd=1234)
     - [PyTorch Official](https://download.pytorch.org/models/resnet50-19c8e357.pth)
   
   - **R101 DCN FCOS3D**: `r101_dcn_fcos3d_pretrain.pth`
     - [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/r101_dcn_fcos3d_pretrain.pth)
     - [Baidu Cloud](https://pan.baidu.com/s/1o7owaQ5G66xqq2S0TldwXQ?pwd=1234)

3. Place downloaded files in the `ckpts/` directory

### Verify Installation

Test that the installation was successful:
```bash
python -c "import mmcv; import torch; print(f'MMCV installed, CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### CUDA Extension Build Errors
- Ensure GCC 9.4 is installed and active
- Check CUDA toolkit installation: `nvcc --version`
- Set environment variables if needed:
  ```bash
  export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
  export PATH=$CUDA_HOME/bin:$PATH
  ```

### Import Errors
- Verify PyTorch CUDA version matches your CUDA toolkit:
  ```python
  import torch
  print(torch.version.cuda)  # Should match your CUDA version
  ```

### Dependency Conflicts
- The package automatically selects compatible scipy versions based on Python version
- numpy==1.22.4 is compatible with both Python 3.8 and 3.10

## Package Structure

All dependencies are managed through `pyproject.toml`:
- Core dependencies are installed by default
- Optional dependencies can be installed via extras:
  - `[all]`: Complete installation with all features
  - `[dev]`: Development tools
  - `[test]`: Testing tools
  - `[vision]`: Vision-related packages
  - Individual groups can be combined: `pip install -e ".[dev,test]"`

## Next Steps

After installation, refer to:
- [Data Preparation Guide](../README.md#data-preparation)
- [Training Documentation](../README.md#training)
- [Evaluation Guide](../README.md#evaluation)