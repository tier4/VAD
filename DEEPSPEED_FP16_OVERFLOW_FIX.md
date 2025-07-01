# DeepSpeed FP16 Overflow Fix

## Summary of Fixes Applied

This document describes the comprehensive fixes applied to resolve the persistent DeepSpeed FP16 overflow errors during VAD model training.

### Root Causes Identified

1. **Incorrect NaN handling** in distance calculations
2. **Missing epsilon protection** in division operations
3. **Unprotected norm operations** that overflow with large FP16 values
4. **Dtype-unaware epsilon values** causing precision issues
5. **Exponential operations** without input clamping
6. **Aggressive loss scaling** in DeepSpeed configuration

### Fixes Applied

#### 1. VAD_head.py
- Fixed NaN handling in `get_best_fut_preds` and `get_traj_cls_target` methods
- Replaced `dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0` with `torch.nan_to_num()`

#### 2. plan_loss.py
- Added `safe_norm()` helper function with clamping to prevent overflow
- Fixed dtype-aware epsilon in `plan_map_dir_loss` atan2 operations
- Replaced all `torch.linalg.norm` calls with `safe_norm`
- Already fixed `segments_intersect` NaN handling in previous attempt

#### 3. CD_loss.py
- Updated epsilon to use dtype-aware value: `torch.finfo(loss_src.dtype).eps`
- Ensures FP16 compatibility in Chamfer Distance calculations

#### 4. functional.py
- Added clamping to `bivariate_gaussian_activation` before exponential
- Prevents exp() overflow with large inputs in FP16

#### 5. ds_config_zero2.json
- Reduced initial_scale_power from 16 to 12 (initial loss scale: 4096 instead of 65536)
- Reduced loss_scale_window from 1000 to 500 (faster adaptation)
- Increased hysteresis from 2 to 4 (more stable scaling)

#### 6. gradient_monitor.py (New)
- Created monitoring hook to diagnose gradient issues
- Logs maximum gradient norms and detects NaN/Inf
- Can enable PyTorch anomaly detection for debugging

## Usage

### Enable Gradient Monitoring

Add to your config file:

```python
custom_hooks = [
    dict(
        type='GradientMonitorHook',
        interval=50,  # Log every 50 iterations
        monitor_grads=True,
        detect_anomaly=False  # Set True for debugging
    )
]
```

### Running Training

The fixes are now integrated into the codebase. Simply run training as usual:

```bash
python -m torch.distributed.run --nproc_per_node=8 tools/train.py configs/VAD/VAD_base_e2e_deepspeed.py --deepspeed configs/deepspeed/ds_config_zero2.json
```

## Verification

Monitor the training logs for:
1. No more "OVERFLOW!" messages
2. Stable loss values
3. Gradient norms within reasonable range (< 1000)

If issues persist, enable the gradient monitor with `detect_anomaly=True` to identify the exact operation causing problems.