#!/usr/bin/env python
"""
Utility script to convert between PyTorch and DeepSpeed checkpoint formats.

Usage:
    # Convert PyTorch checkpoint to DeepSpeed format
    python tools/convert_checkpoint_deepspeed.py \
        --input path/to/pytorch_checkpoint.pth \
        --output path/to/deepspeed_checkpoint/ \
        --mode pytorch_to_deepspeed

    # Convert DeepSpeed checkpoint to PyTorch format
    python tools/convert_checkpoint_deepspeed.py \
        --input path/to/deepspeed_checkpoint/ \
        --output path/to/pytorch_checkpoint.pth \
        --mode deepspeed_to_pytorch
"""

import argparse
import os
import torch
import json
from collections import OrderedDict


def convert_pytorch_to_deepspeed(pytorch_ckpt_path, output_dir):
    """Convert standard PyTorch checkpoint to DeepSpeed format."""
    print(f"Converting PyTorch checkpoint: {pytorch_ckpt_path}")
    print(f"Output directory: {output_dir}")
    
    # Load PyTorch checkpoint
    ckpt = torch.load(pytorch_ckpt_path, map_location='cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model state dict
    if 'state_dict' in ckpt:
        model_state = ckpt['state_dict']
    elif 'model' in ckpt:
        model_state = ckpt['model']
    else:
        # Assume the checkpoint is the state dict itself
        model_state = ckpt
    
    # Create DeepSpeed checkpoint structure
    # For ZeRO-2, we need to save model states
    model_states = {'module': model_state}
    
    # Save model states
    model_path = os.path.join(output_dir, 'mp_rank_00_model_states.pt')
    torch.save(model_states, model_path)
    print(f"Saved model states to: {model_path}")
    
    # Create placeholder optimizer states (empty for conversion)
    optim_states = {
        'optimizer_state_dict': {},
        'param_slice_mappings': {},
        'lr_scheduler': None
    }
    
    # Save optimizer states
    optim_path = os.path.join(output_dir, 'zero_pp_rank_0_mp_rank_00_optim_states.pt')
    torch.save(optim_states, optim_path)
    print(f"Saved optimizer states to: {optim_path}")
    
    # Save metadata
    metadata = {
        'epoch': ckpt.get('epoch', 0),
        'iter': ckpt.get('iter', 0),
        'global_step': ckpt.get('global_step', 0)
    }
    
    meta_path = os.path.join(output_dir, 'meta.pth')
    torch.save(metadata, meta_path)
    print(f"Saved metadata to: {meta_path}")
    
    # Create DeepSpeed config file for reference
    ds_config = {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 0.0001,
                "weight_decay": 0.01
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        }
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(ds_config, f, indent=2)
    print(f"Saved config to: {config_path}")
    
    print("Conversion complete!")


def convert_deepspeed_to_pytorch(deepspeed_ckpt_dir, output_path):
    """Convert DeepSpeed checkpoint to standard PyTorch format."""
    print(f"Converting DeepSpeed checkpoint: {deepspeed_ckpt_dir}")
    print(f"Output path: {output_path}")
    
    # Check if directory exists
    if not os.path.isdir(deepspeed_ckpt_dir):
        raise ValueError(f"DeepSpeed checkpoint directory not found: {deepspeed_ckpt_dir}")
    
    # Load model states
    model_path = os.path.join(deepspeed_ckpt_dir, 'mp_rank_00_model_states.pt')
    if not os.path.exists(model_path):
        # Try alternative naming
        model_files = [f for f in os.listdir(deepspeed_ckpt_dir) if 'model_states' in f]
        if model_files:
            model_path = os.path.join(deepspeed_ckpt_dir, model_files[0])
        else:
            raise FileNotFoundError(f"Model states not found in {deepspeed_ckpt_dir}")
    
    model_states = torch.load(model_path, map_location='cpu')
    
    # Extract state dict
    if 'module' in model_states:
        state_dict = model_states['module']
    else:
        state_dict = model_states
    
    # Remove 'module.' prefix if present (from DDP)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
    
    # Load metadata if available
    meta_path = os.path.join(deepspeed_ckpt_dir, 'meta.pth')
    if os.path.exists(meta_path):
        metadata = torch.load(meta_path, map_location='cpu')
    else:
        metadata = {}
    
    # Create PyTorch checkpoint
    checkpoint = {
        'state_dict': new_state_dict,
        'epoch': metadata.get('epoch', 0),
        'iter': metadata.get('iter', 0)
    }
    
    # Save PyTorch checkpoint
    torch.save(checkpoint, output_path)
    print(f"Saved PyTorch checkpoint to: {output_path}")
    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Convert between PyTorch and DeepSpeed checkpoint formats'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input checkpoint path (file for PyTorch, directory for DeepSpeed)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output checkpoint path (directory for DeepSpeed, file for PyTorch)'
    )
    parser.add_argument(
        '--mode',
        required=True,
        choices=['pytorch_to_deepspeed', 'deepspeed_to_pytorch'],
        help='Conversion mode'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'pytorch_to_deepspeed':
        convert_pytorch_to_deepspeed(args.input, args.output)
    else:
        convert_deepspeed_to_pytorch(args.input, args.output)


if __name__ == '__main__':
    main()