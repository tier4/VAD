#!/usr/bin/env python
"""Debug optimizer construction for DeepSpeed."""

import torch
import torch.nn as nn
from mmcv import Config
from mmcv.models import build_model
from mmcv.optims import build_optimizer

# Load the config
cfg = Config.fromfile('configs/VAD/VAD_base_e2e_deepspeed.py')

# Create a simple dummy model for testing
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        # Add some params that might not require grad
        self.register_buffer('buffer', torch.ones(10))
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Build model
model = DummyModel()
print(f"Model has {sum(p.numel() for p in model.parameters())} total parameters")
print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

# Build optimizer
optimizer = build_optimizer(model, cfg.optimizer)

print(f"\nOptimizer has {len(optimizer.param_groups)} parameter groups")
for i, pg in enumerate(optimizer.param_groups):
    print(f"Parameter group {i}: {len(pg['params'])} parameters")
    if len(pg['params']) > 0:
        print(f"  - Learning rate: {pg['lr']}")
        print(f"  - Weight decay: {pg.get('weight_decay', 0)}")
        param_names = []
        for p in pg['params']:
            # Find parameter name
            for name, param in model.named_parameters():
                if param is p:
                    param_names.append(name)
                    break
        print(f"  - Parameters: {param_names[:5]}{'...' if len(param_names) > 5 else ''}")

# Filter empty param groups
optimizer.param_groups = [pg for pg in optimizer.param_groups if len(pg['params']) > 0]
print(f"\nAfter filtering: {len(optimizer.param_groups)} parameter groups")

# Check if there are still empty groups
for i, pg in enumerate(optimizer.param_groups):
    if len(pg['params']) == 0:
        print(f"WARNING: Parameter group {i} is still empty!")

# Test with actual VAD model
print("\n" + "="*50)
print("Testing with actual VAD model...")
try:
    vad_model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    print(f"VAD model built successfully")
    print(f"VAD model has {sum(p.numel() for p in vad_model.parameters())} total parameters")
    print(f"VAD model has {sum(p.numel() for p in vad_model.parameters() if p.requires_grad)} trainable parameters")
    
    # Build optimizer for VAD model
    vad_optimizer = build_optimizer(vad_model, cfg.optimizer)
    print(f"\nVAD optimizer has {len(vad_optimizer.param_groups)} parameter groups")
    for i, pg in enumerate(vad_optimizer.param_groups):
        print(f"Parameter group {i}: {len(pg['params'])} parameters")
except Exception as e:
    print(f"Error building VAD model: {e}")
    import traceback
    traceback.print_exc()