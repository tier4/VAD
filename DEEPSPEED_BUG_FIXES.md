# DeepSpeed Integration Bug Fixes

This document summarizes the bugs that were fixed during DeepSpeed integration with the VAD training pipeline.

## Bug 1: TypeError with train_batch_size

### Error
```
TypeError: '>' not supported between instances of 'str' and 'int'
```

### Root Cause
- The DeepSpeed config had `train_batch_size: "auto"` which is a string
- DeepSpeed was trying to validate `train_batch > 0` causing a type comparison error

### Fix
In `mmcv/mmdet3d_plugin/bevformer/apis/mmdet_train.py`, calculate the total batch size:
```python
# Calculate total train_batch_size
world_size = dist.get_world_size() if dist.is_initialized() else 1
train_batch_size = cfg.data.samples_per_gpu * world_size * ds_config['gradient_accumulation_steps']
ds_config['train_batch_size'] = train_batch_size
```

## Bug 2: RuntimeError with empty tensor list

### Error
```
RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

### Root Cause
- The MMCV optimizer constructor was creating a separate parameter group for EACH parameter
- This resulted in 900+ parameter groups for the VAD model
- Some parameter groups were empty after filtering
- DeepSpeed's ZeRO optimizer couldn't handle the complex parameter grouping

### Fix
In `mmcv/mmdet3d_plugin/bevformer/apis/mmdet_train.py`, build a simplified optimizer for DeepSpeed:
```python
# For DeepSpeed, we need a simpler optimizer configuration
# Build a simplified optimizer for DeepSpeed
optimizer_cfg = cfg.optimizer.copy()
optimizer_cfg.pop('paramwise_cfg', None)  # Remove paramwise config

# Get all parameters that require gradients
params = [p for p in model.parameters() if p.requires_grad]

# Build optimizer with single parameter group
optimizer_type = optimizer_cfg.pop('type')
optimizer_class = getattr(torch.optim, optimizer_type)
optimizer = optimizer_class(params, **optimizer_cfg)
```

Also simplified the DeepSpeed config in `configs/VAD/VAD_base_e2e_deepspeed.py`:
```python
optimizer = dict(
    type='AdamW',
    lr=4e-4,
    weight_decay=0.01
)
```

## Bug 3: Distributed Training Detection

### Error
- Training was showing "Distributed training: False" when launched with DeepSpeed

### Fix
In `tools/train.py`, detect DeepSpeed's distributed environment:
```python
# When using DeepSpeed launcher, LOCAL_RANK is set automatically
if args.launcher == 'none' and 'LOCAL_RANK' not in os.environ:
    distributed = False
else:
    distributed = True
    # If using DeepSpeed launcher, init with pytorch backend
    if args.launcher == 'none' and 'LOCAL_RANK' in os.environ:
        init_dist('pytorch', **cfg.dist_params)
```

## Bug 4: DeepSpeed Optimizer Type Error

### Error
```
TypeError: EpochBasedRunner: optimizer must be a torch.optim.Optimizer object or dict or None, but got <class 'deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer'>
```

### Root Cause
- After DeepSpeed initialization, the optimizer becomes a `DeepSpeedZeroOptimizer`
- MMCV's BaseRunner expects a standard PyTorch optimizer or None
- The DeepSpeed engine wraps both model and optimizer

### Fix
In `mmcv/mmdet3d_plugin/bevformer/apis/mmdet_train.py`, pass None as optimizer to the runner when using DeepSpeed:
```python
# When using DeepSpeed, pass None as optimizer to the runner
# DeepSpeed engine handles optimization internally
runner_optimizer = None if using_deepspeed else optimizer
```

## Bug 5: DeepSpeedOptimizerHook Parameter Error

### Error
```
TypeError: __init__() got an unexpected keyword argument 'grad_clip'
```

### Root Cause
- The optimizer config was passing `grad_clip` parameter to DeepSpeedOptimizerHook
- The hook's `__init__` method didn't accept parameters

### Fix
In `mmcv/runner/hooks/optimizer.py`, accept parameters for compatibility:
```python
def __init__(self, grad_clip=None, **kwargs):
    # Accept grad_clip for compatibility but ignore it - handled by DeepSpeed config
    super().__init__(grad_clip=None)
```

## Summary

The main issues were:
1. Type mismatch in batch size configuration
2. Optimizer parameter grouping incompatibility with DeepSpeed
3. Distributed environment detection
4. Runner expecting PyTorch optimizer instead of DeepSpeed optimizer
5. Hook parameter compatibility

All bugs have been fixed and DeepSpeed integration is now working properly with:
- Correct batch size calculation
- Simplified optimizer with single parameter group
- Proper distributed training detection
- Correct optimizer handling in the runner
- Compatible hook initialization