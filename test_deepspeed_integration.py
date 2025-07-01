#!/usr/bin/env python
"""Test script to verify DeepSpeed integration is working correctly."""

import torch
import torch.nn as nn
import torch.distributed as dist
import os
import json
import deepspeed

class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def test_deepspeed():
    """Test DeepSpeed initialization and basic operations."""
    # Initialize distributed environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group('nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank}/{world_size}: Initialized distributed environment")
    
    # Create model
    model = DummyModel()
    
    # DeepSpeed config
    ds_config = {
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_clipping": 35.0,
        "wall_clock_breakdown": False
    }
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    print(f"Rank {rank}: DeepSpeed initialized successfully")
    
    # Test forward and backward pass
    batch_size = 8
    input_data = torch.randn(batch_size, 10).cuda().half()  # Convert to FP16
    target = torch.randn(batch_size, 1).cuda().half()  # Convert to FP16
    
    # Forward pass
    output = model_engine(input_data)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    model_engine.backward(loss)
    
    # Optimizer step
    model_engine.step()
    
    print(f"Rank {rank}: Forward/backward pass completed successfully")
    print(f"Rank {rank}: Loss = {loss.item():.4f}")
    
    # Test checkpoint saving (only rank 0)
    if rank == 0:
        ckpt_dir = "./test_deepspeed_checkpoint"
        model_engine.save_checkpoint(ckpt_dir, tag="test")
        print(f"Rank {rank}: Checkpoint saved to {ckpt_dir}")
        
        # Clean up
        import shutil
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
    
    dist.barrier()
    print(f"Rank {rank}: Test completed successfully!")

if __name__ == "__main__":
    test_deepspeed()