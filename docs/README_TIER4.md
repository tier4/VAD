# Tier4 Coordinate System for VAD

## Overview

The Tier4 coordinate system is an alternative coordinate representation designed for autonomous driving applications, where the origin is placed at the **center of the rear axis** rather than at the LiDAR sensor position. This implementation provides Tier4 coordinate support for both nuScenes and CARLA/Bench2Drive datasets in the VAD framework.

## Key Features

### Coordinate Transformation
The Tier4 system applies a **90-degree counter-clockwise rotation** to the standard coordinate system:
- **Y-axis → X-axis** (lateral becomes longitudinal)
- **-X-axis → Y-axis** (negative longitudinal becomes lateral)  
- **Z-axis remains unchanged**

### Origin Position
- **Standard System**: Origin at LiDAR sensor (typically roof-mounted)
- **Tier4 System**: Origin at rear axis center (closer to ground)
- **Z-range Impact**: Lower Z values (-0.16 to 3.84m) due to lower origin point

## Mathematical Transformations

### Rotation Matrix
```python
[[0,  1, 0],   # Y → X
 [-1, 0, 0],   # -X → Y
 [0,  0, 1]]   # Z → Z
```

### Transformation Formulas
- **Position**: `(x_new, y_new) = (y_old, -x_old)`
- **Velocity**: `(vx_new, vy_new) = (vy_old, -vx_old)`
- **Yaw Angle**: `yaw_new = yaw_old - π/2` (normalized to [-π, π])

## Available Configurations

### nuScenes Dataset
- **Standard**: `configs/VAD/VAD_tiny_e2e.py`
- **Tier4**: `configs/VAD/VAD_tiny_e2e_tier4.py`

### CARLA/Bench2Drive Dataset
- **Standard**: `configs/VAD/VAD_tiny_carla.py`
- **Tier4**: `configs/VAD/VAD_tiny_carla_tier4.py` ✨ NEW

## Training with Tier4 Coordinates

### Single GPU Training

```bash
# CARLA/Bench2Drive with Tier4 coordinates
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/VAD/VAD_tiny_carla_tier4.py \
    --launcher none \
    --work-dir work_dirs/vad_tiny_carla_tier4

# nuScenes with Tier4 coordinates  
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/VAD/VAD_tiny_e2e_tier4.py \
    --launcher none \
    --work-dir work_dirs/vad_tiny_tier4
```

### Multi-GPU Training

```bash
# 8 GPUs for CARLA/Bench2Drive Tier4
python -m torch.distributed.run --nproc_per_node=8 --master_port=2333 \
    tools/train.py \
    configs/VAD/VAD_tiny_carla_tier4.py \
    --launcher pytorch \
    --deterministic \
    --work-dir work_dirs/vad_carla_tier4_8gpu

# Alternative using distribution script
./tools/dist_train.sh configs/VAD/VAD_tiny_carla_tier4.py 8
```

### Resume Training

```bash
# Resume from checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/VAD/VAD_tiny_carla_tier4.py \
    --launcher none \
    --resume-from work_dirs/vad_tiny_carla_tier4/latest.pth \
    --work-dir work_dirs/vad_tiny_carla_tier4
```

## Testing and Evaluation

### Important: Single GPU Testing Only
Multi-GPU testing produces incorrect results. Always use single GPU for evaluation:

```bash
# Test CARLA/Bench2Drive Tier4 model
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/VAD/VAD_tiny_carla_tier4.py \
    work_dirs/vad_tiny_carla_tier4/latest.pth \
    --launcher none \
    --eval bbox \
    --tmpdir tmp_tier4

# Test nuScenes Tier4 model
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/VAD/VAD_tiny_e2e_tier4.py \
    work_dirs/vad_tiny_tier4/latest.pth \
    --launcher none \
    --eval bbox \
    --tmpdir tmp_tier4
```

## Visualization

### Tier4-Aware Visualization

```bash
# Visualize results with Tier4 coordinates
python tools/analysis_tools/visualization_t4.py \
    --config configs/VAD/VAD_tiny_carla_tier4.py \
    --result-path work_dirs/vad_tiny_carla_tier4/results.pkl \
    --save-path ./vis_tier4_output
```

## Configuration Details

### Key Parameter Changes (CARLA/B2D)

| Parameter | Standard | Tier4 | Notes |
|-----------|----------|-------|-------|
| **point_cloud_range** | [-15, -30, -2, 15, 30, 2] | [-30, -16, -0.16, 30, 16, 3.84] | Rotated & lower Z |
| **detection_post_range** | [-20, -35, -10, 20, 35, 10] | [-35, -21, -0.66, 35, 21, 4.34] | 5m XY padding |
| **grid_size** | [50, 100, 1] | [100, 53, 1] | Swapped X/Y dimensions |
| **bev_h × bev_w** | 100 × 50 | 53 × 100 | Rotated BEV grid |
| **dataset_type** | B2D_VAD_Dataset | B2D_VAD_DatasetTier4 | Tier4 transformations |

### Dataset Class Hierarchy

```
B2D_VAD_Dataset (Standard coordinates)
    └── B2D_VAD_DatasetTier4 (Applies Tier4 transformations)
        - Inherits all B2D functionality
        - Overrides get_data_info() for coordinate transformation
        - Transforms: bboxes, trajectories, map polylines
```

## Implementation Details

### Files Created/Modified

1. **`mmcv/datasets/B2D_vad_dataset_tier4.py`**
   - New dataset class with Tier4 transformations
   - Static methods for coordinate conversions
   - Overridden `get_data_info()` method

2. **`configs/VAD/VAD_tiny_carla_tier4.py`**
   - Complete configuration for CARLA/B2D with Tier4
   - Adjusted ranges and grid dimensions
   - Uses B2D_VAD_DatasetTier4

3. **`mmcv/datasets/__init__.py`**
   - Added B2D_VAD_DatasetTier4 import

### Coordinate Transformation Process

The transformation is applied automatically during data loading:

1. **Data Loading**: Base dataset loads raw CARLA/B2D data
2. **Transformation**: `get_data_info()` applies Tier4 transforms to:
   - Object bounding boxes (position, velocity, yaw)
   - Ego vehicle trajectories (past and future)
   - Map polylines and lane markings
3. **Model Input**: Transformed data fed to model in Tier4 coordinates
4. **Output**: Predictions are in Tier4 coordinate system

## Benefits of Tier4 System

1. **Vehicle-Centric**: Origin at rear axis is more natural for vehicle control
2. **Planning Alignment**: Better aligns with vehicle dynamics and control systems
3. **Industry Standard**: Used by Tier IV autonomous driving company
4. **Lower Origin**: Accounts for actual vehicle sensor mounting position

## Troubleshooting

### Common Issues

1. **Import Error for B2D_VAD_DatasetTier4**
   - Ensure `mmcv/datasets/__init__.py` includes the import
   - Rebuild package: `pip install -v -e .`

2. **Incorrect Grid Dimensions**
   - Verify point_cloud_range matches expected Tier4 values
   - Check voxel_size is consistent

3. **Visualization Misalignment**
   - Use `visualization_t4.py` for Tier4 configs
   - Standard visualization assumes different coordinate system

4. **Training Convergence Issues**
   - May need to adjust learning rate (currently 4e-4)
   - Ensure data augmentation respects coordinate system

### Verification Commands

```bash
# Verify dataset registration
python -c "from mmcv.datasets import B2D_VAD_DatasetTier4; print('✓ Import successful')"

# Test coordinate transformations
python -c "
from mmcv.datasets import B2D_VAD_DatasetTier4
import numpy as np
pos = np.array([10, 20])
new_pos = B2D_VAD_DatasetTier4.transform_position_to_tier4(pos)
print(f'Position {pos} → {new_pos}')  # Should output [20, -10]
"

# Check config loading
python -c "
from mmcv import Config
cfg = Config.fromfile('configs/VAD/VAD_tiny_carla_tier4.py')
print(f'Dataset: {cfg.dataset_type}')  # Should be B2D_VAD_DatasetTier4
print(f'PC Range: {cfg.point_cloud_range}')
"
```

## Performance Considerations

- **Training Time**: Similar to standard coordinate system
- **Memory Usage**: Identical to standard configuration
- **Accuracy**: May require fine-tuning for optimal results
- **Inference Speed**: No performance impact

## Future Enhancements

Potential improvements for Tier4 support:

1. **Automatic Conversion**: Tools to convert existing checkpoints between coordinate systems
2. **Mixed Training**: Support for training with both coordinate systems
3. **Visualization Tools**: Enhanced visualization for Tier4-specific features
4. **Evaluation Metrics**: Tier4-aware evaluation metrics

## References

- [Tier IV Company](https://tier4.jp/) - Original developers of Tier4 coordinate system
- [VAD Paper](https://github.com/hustvl/VAD) - Base VAD implementation
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) - CARLA dataset source

## Support

For issues specific to Tier4 coordinate system:
1. Check this documentation first
2. Verify coordinate transformations with test commands
3. Compare with standard coordinate configs for reference
4. Report issues with `[Tier4]` tag in issue title

---

*Last Updated: 2025*  
*VAD Tier4 Implementation v1.0*