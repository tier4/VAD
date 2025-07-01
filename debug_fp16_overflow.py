#!/usr/bin/env python
"""
Debug script to identify FP16 overflow sources in VAD model.
Run this to test individual components and find which operations cause overflow.
"""

import torch
import torch.nn as nn
import numpy as np

# Define safe_norm locally to avoid import issues
def safe_norm(x, y=None, dim=-1, clamp_range=(-100, 100)):
    """Compute torch.linalg.norm with clamping to prevent FP16 overflow."""
    if y is not None:
        x = x - y
    x = torch.clamp(x, min=clamp_range[0], max=clamp_range[1])
    return torch.linalg.norm(x, dim=dim)

def test_safe_norm():
    """Test if safe_norm function handles extreme values properly."""
    print("Testing safe_norm function...")
    
    # Test with extreme values
    x = torch.tensor([[1e5, 1e5], [1e-5, 1e-5]], dtype=torch.float16)
    y = torch.tensor([[1e5, 1e5], [1e-5, 1e-5]], dtype=torch.float16)
    
    try:
        result = safe_norm(x, y, dim=-1)
        print(f"safe_norm result: {result}")
        print(f"Contains inf: {torch.isinf(result).any()}")
        print(f"Contains nan: {torch.isnan(result).any()}")
    except Exception as e:
        print(f"safe_norm failed: {e}")

def test_torch_operations():
    """Test various torch operations that might cause overflow."""
    print("\nTesting torch operations in FP16...")
    
    # Test exponential
    x = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float16)
    print(f"Input: {x}")
    try:
        exp_x = torch.exp(x)
        print(f"torch.exp: {exp_x}")
    except Exception as e:
        print(f"torch.exp failed: {e}")
    
    # Test with clamping
    x_clamped = torch.clamp(x, min=-10, max=10)
    exp_x_clamped = torch.exp(x_clamped)
    print(f"torch.exp with clamping: {exp_x_clamped}")
    
    # Test division by small numbers
    eps_fp16 = torch.finfo(torch.float16).eps
    eps_fp32 = torch.finfo(torch.float32).eps
    print(f"\nFP16 epsilon: {eps_fp16}")
    print(f"FP32 epsilon: {eps_fp32}")
    
    small_val = torch.tensor(1e-8, dtype=torch.float16)
    print(f"1 / {small_val} = {1 / (small_val + eps_fp16)}")
    
    # Test linalg.norm with large values
    large_vec = torch.tensor([[1e4, 1e4], [1e3, 1e3]], dtype=torch.float16)
    print(f"\nLarge vector: {large_vec}")
    norm = torch.linalg.norm(large_vec, dim=-1)
    print(f"Norm: {norm}")
    print(f"Contains inf: {torch.isinf(norm).any()}")

def test_loss_scale_limits():
    """Test what happens at different loss scales."""
    print("\nTesting loss scaling behavior...")
    
    # Simulate a loss value
    loss = torch.tensor(100.0, dtype=torch.float16)
    
    for scale_power in [16, 12, 8, 4, 0]:
        scale = 2 ** scale_power
        scaled_loss = loss * scale
        print(f"Loss scale 2^{scale_power} = {scale}: scaled_loss = {scaled_loss}")
        print(f"  Is inf: {torch.isinf(scaled_loss)}")
        print(f"  Is nan: {torch.isnan(scaled_loss)}")

def test_attention_scores():
    """Test attention score computation which often causes overflow."""
    print("\nTesting attention scores...")
    
    # Simulate attention scores
    dim = 256
    seq_len = 100
    
    q = torch.randn(1, seq_len, dim, dtype=torch.float16) * 0.1
    k = torch.randn(1, seq_len, dim, dtype=torch.float16) * 0.1
    
    # Standard attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(dim)
    print(f"Attention scores shape: {scores.shape}")
    print(f"Scores max: {scores.max()}, min: {scores.min()}")
    print(f"Contains inf: {torch.isinf(scores).any()}")
    
    # Apply softmax
    attention = torch.softmax(scores, dim=-1)
    print(f"After softmax - contains nan: {torch.isnan(attention).any()}")

def check_model_initialization():
    """Check if model weights are initialized properly for FP16."""
    print("\nChecking typical weight initialization ranges...")
    
    # Simulate different initialization methods
    linear = nn.Linear(256, 256)
    
    print(f"Linear weight max: {linear.weight.max()}, min: {linear.weight.min()}")
    print(f"Linear weight std: {linear.weight.std()}")
    
    # Convert to FP16 and check
    linear_fp16 = linear.half()
    print(f"FP16 weight max: {linear_fp16.weight.max()}, min: {linear_fp16.weight.min()}")

def main():
    print("=== FP16 Overflow Debugging ===")
    print(f"FP16 range: [{torch.finfo(torch.float16).min}, {torch.finfo(torch.float16).max}]")
    print(f"FP16 smallest normal: {torch.finfo(torch.float16).tiny}")
    
    test_safe_norm()
    test_torch_operations()
    test_loss_scale_limits()
    test_attention_scores()
    check_model_initialization()
    
    print("\n=== Recommendations ===")
    print("1. Initial loss scale should be much lower (2^4 or 2^8)")
    print("2. All norm operations need aggressive clamping")
    print("3. Attention scores may need temperature scaling")
    print("4. Consider gradient accumulation to reduce per-step gradients")

if __name__ == "__main__":
    main()