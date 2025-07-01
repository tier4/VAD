#!/bin/bash
# Setup environment for optimal DeepSpeed performance with VAD training

echo "Setting up DeepSpeed environment for VAD training..."

# NCCL settings for better multi-GPU communication
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# CPU affinity for better performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# DeepSpeed settings
export DS_ACCELERATOR="cuda"

# PyTorch settings for better performance
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"  # Cover common GPU architectures
export CUDA_LAUNCH_BLOCKING=0

# Disable TF32 for better numerical precision (optional)
# export NVIDIA_TF32_OVERRIDE=0

# Optional: For debugging DeepSpeed
# export DEEPSPEED_LOG_LEVEL=DEBUG
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Print environment info
echo "Environment setup complete!"
echo "NCCL_DEBUG=$NCCL_DEBUG"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "DS_ACCELERATOR=$DS_ACCELERATOR"

# Check if DeepSpeed is installed
if python -c "import deepspeed" &> /dev/null; then
    echo "DeepSpeed is installed. Version:"
    python -c "import deepspeed; print(f'  {deepspeed.__version__}')"
else
    echo "WARNING: DeepSpeed is not installed. Please run: pip install deepspeed>=0.12.0"
fi

# Check CUDA availability
if python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
    echo "CUDA is available. GPU count:"
    python -c "import torch; print(f'  {torch.cuda.device_count()}')"
else
    echo "ERROR: CUDA is not available. Please check your CUDA installation."
fi