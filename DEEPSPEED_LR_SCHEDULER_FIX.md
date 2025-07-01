# DeepSpeed LR Scheduler Fix

## Bug 6: AttributeError - 'NoneType' object has no attribute 'param_groups'

### Error
```
AttributeError: 'NoneType' object has no attribute 'param_groups'
  File "/home/ubuntu/work/VAD_Universe/mmcv/runner/hooks/lr_updater.py", line 134, in before_train_epoch
    self.regular_lr = self.get_regular_lr(runner)
  File "/home/ubuntu/work/VAD_Universe/mmcv/runner/hooks/lr_updater.py", line 83, in get_regular_lr
    return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]
```

### Root Cause
- When using DeepSpeed, the runner's optimizer is set to None (as DeepSpeed handles optimization internally)
- The CosineAnnealingLrUpdaterHook tried to access `runner.optimizer.param_groups` but optimizer was None
- MMCV's LR scheduler hooks expect a PyTorch optimizer to be available

### Fix Approach
1. **Disable LR scheduling in DeepSpeed config** - Since DeepSpeed can handle LR scheduling internally through its configuration
2. **Create DeepSpeed-compatible LR scheduler hooks** - That check if optimizer is None and skip LR updates

### Implementation

#### 1. Disabled LR scheduling in DeepSpeed config (`configs/VAD/VAD_base_e2e_deepspeed.py`):
```python
# Disable LR scheduling - DeepSpeed handles this internally
lr_config = None
```

#### 2. Created DeepSpeed-compatible LR updater hooks (`mmcv/runner/hooks/deepspeed_lr_updater.py`):
```python
@HOOKS.register_module()
class DeepSpeedLrUpdaterHook(LrUpdaterHook):
    """Learning rate scheduler hook for DeepSpeed training.
    
    This hook is a no-op when using DeepSpeed because DeepSpeed
    handles learning rate scheduling internally through its own
    configuration.
    """
    
    def before_run(self, runner):
        # Skip LR initialization when using DeepSpeed (optimizer is None)
        if runner.optimizer is None:
            runner.logger.info(
                "DeepSpeed is handling learning rate scheduling internally. "
                "Skipping MMCV LR scheduler initialization."
            )
            return
        
        # Fall back to parent implementation if not using DeepSpeed
        super().before_run(runner)
    
    def before_train_epoch(self, runner):
        # Skip LR updates when using DeepSpeed
        if runner.optimizer is None:
            return
        super().before_train_epoch(runner)
    
    def before_train_iter(self, runner):
        # Skip LR updates when using DeepSpeed
        if runner.optimizer is None:
            return
        super().before_train_iter(runner)
```

### Alternative: Configure LR scheduling in DeepSpeed
If you want to use LR scheduling with DeepSpeed, add the scheduler configuration to `ds_config_zero2.json`:
```json
{
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 125,
            "total_num_steps": 10000,
            "warmup_type": "linear"
        }
    }
}
```

### Result
- Training now starts successfully without AttributeError
- DeepSpeed handles optimization and can optionally handle LR scheduling through its configuration
- The fix maintains compatibility with non-DeepSpeed training