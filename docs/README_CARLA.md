# VAD Model for CARLA/Bench2Drive Dataset

This document describes the integration of the VAD (Vectorized Scene Representation for Efficient Autonomous Driving) model with CARLA/Bench2Drive dataset.

## Overview

The VAD model has been integrated to work with CARLA simulator data through the Bench2Drive dataset format. This integration includes:
- Support for CARLA's object classes
- Map element detection (lanes, traffic lights, stop signs, etc.)
- End-to-end trajectory planning
- Temporal modeling with multi-frame sequences

## Dataset Structure

The CARLA/Bench2Drive data should be organized as follows:
```
data_b2d/
├── bench2drive/          # Raw data (symlink to actual data location)
├── infos/
│   ├── b2d_infos_train.pkl
│   ├── b2d_infos_val.pkl
│   └── b2d_map_infos.pkl
└── splits/
```

## Configuration

The main configuration file is located at: `configs/VAD/VAD_tiny_carla.py`

### Key Features:
- **Object Classes**: car, van, truck, bicycle, traffic_sign, traffic_cone, traffic_light, pedestrian, others
- **Map Classes**: Broken, Solid, SolidSolid, Center, TrafficLight, StopSign
- **Temporal Modeling**: 3 frames queue length with 5 frame intervals
- **Planning Horizon**: 6 future frames
- **Point Cloud Range**: [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

## Training

### Single GPU Training
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/VAD/VAD_tiny_carla.py \
    --launcher none \
    --work-dir work_dirs/vad_tiny_carla \
    --deterministic
```

### Multi-GPU Training (Recommended)
```bash
# Using 8 GPUs (adjust nproc_per_node as needed)
python -m torch.distributed.run --nproc_per_node=8 --master_port=29500 \
    tools/train.py configs/VAD/VAD_tiny_carla.py \
    --launcher pytorch \
    --work-dir work_dirs/vad_tiny_carla \
    --deterministic
```

### Alternative Multi-GPU Training
```bash
# Using the distributed training script (adjust the number of GPUs)
./tools/dist_train.sh configs/VAD/VAD_tiny_carla.py 8 \
    --work-dir work_dirs/vad_tiny_carla
```

### Resume Training from Checkpoint
```bash
python tools/train.py configs/VAD/VAD_tiny_carla.py \
    --resume-from work_dirs/vad_tiny_carla/latest.pth \
    --launcher none \
    --work-dir work_dirs/vad_tiny_carla
```

## Testing/Evaluation

### Single GPU Testing (REQUIRED for accurate metrics)
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/VAD/VAD_tiny_carla.py \
    work_dirs/vad_tiny_carla/latest.pth \
    --launcher none \
    --eval bbox \
    --tmpdir tmp_results
```

### Generate Results for Visualization
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/VAD/VAD_tiny_carla.py \
    work_dirs/vad_tiny_carla/latest.pth \
    --launcher none \
    --format-only \
    --eval-options jsonfile_prefix=work_dirs/vad_tiny_carla/results
```

### Visualization
```bash
python tools/analysis_tools/visualization.py \
    --result-path work_dirs/vad_tiny_carla/results_nusc.json \
    --save-path work_dirs/vad_tiny_carla/vis_results
```

## Training Parameters

### Default Hyperparameters:
- **Batch Size**: 16 samples per GPU
- **Learning Rate**: 8e-4 (AdamW optimizer)
- **Training Epochs**: 60
- **Image Backbone**: ResNet-50 (pretrained)
- **BEV Grid Size**: 117 x 58 (height x width)
- **Number of Queries**: 300 for objects, 100 for map vectors

### Memory Requirements:
- Single GPU training: ~24GB VRAM (reduce batch_size if needed)
- Multi-GPU training: ~16GB VRAM per GPU (with batch_size=16)

### Adjusting for Limited Resources:
If you have limited GPU memory, you can modify the config:
```python
# In configs/VAD/VAD_tiny_carla.py
data = dict(
    samples_per_gpu=8,  # Reduce from 16 to 8
    workers_per_gpu=8,  # Reduce from 16 to 8
    ...
)
```

## Model Outputs

The model provides multiple outputs:
1. **3D Object Detection**: Bounding boxes, classes, and velocities
2. **Map Element Detection**: Vectorized lane lines and traffic elements
3. **Trajectory Prediction**: Future trajectories for detected objects
4. **Ego Planning**: Future trajectory for the ego vehicle

## Evaluation Metrics

### Object Detection Metrics:
- **mAP**: Mean Average Precision
- **mATE**: Mean Average Translation Error
- **mASE**: Mean Average Scale Error
- **mAOE**: Mean Average Orientation Error
- **mAVE**: Mean Average Velocity Error
- **NDS**: NuScenes Detection Score

### Motion Prediction Metrics:
- **EPA**: End-Point Accuracy
- **ADE**: Average Displacement Error
- **FDE**: Final Displacement Error
- **MR**: Miss Rate

### Planning Metrics:
- **L2 Error**: Trajectory prediction error
- **Collision Rate**: Collision with obstacles
- **Map Violation**: Deviation from mapped lanes

## Important Notes

1. **Single GPU for Testing**: Always use single GPU for evaluation to ensure accurate metrics. Multi-GPU testing may produce incorrect results.

2. **Data Preprocessing**: Ensure the CARLA data is properly converted to Bench2Drive format with the correct pickle files.

3. **Pretrained Weights**: Download the ResNet-50 pretrained weights:
   ```bash
   mkdir -p ckpts
   wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpts/resnet50-19c8e357.pth
   ```

4. **Environment Setup**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   pip install -v -e .
   ```

5. **CUDA Extensions**: The project builds custom CUDA operators. Ensure proper CUDA toolkit (11.8) and GCC (9.4) are installed.

## Troubleshooting

### Out of Memory Errors:
- Reduce `samples_per_gpu` in the config
- Reduce `queue_length` (temporal frames)
- Use gradient checkpointing (if implemented)

### Slow Training:
- Reduce `workers_per_gpu` if CPU is bottlenecked
- Ensure data is on fast storage (SSD recommended)
- Use mixed precision training (fp16)

### Poor Performance:
- Ensure proper data augmentation is applied
- Check if pretrained weights are loaded correctly
- Verify the learning rate schedule
- Train for more epochs (default is 60)

## Custom Modifications

To adapt the model for specific CARLA scenarios:

1. **Modify Classes**: Update `class_names` and `name_mapping` in the config
2. **Adjust Detection Range**: Modify `point_cloud_range` and `eval_cfg['class_range']`
3. **Change Temporal Settings**: Adjust `queue_length`, `past_frames`, `future_frames`
4. **Map Elements**: Update `map_classes` for different lane types

## Citation

If you use this code, please cite the original VAD paper:
```bibtex
@inproceedings{jiang2023vad,
  title={VAD: Vectorized Scene Representation for Efficient Autonomous Driving},
  author={Jiang, Bo and Chen, Shaoyu and Xu, Qing and Liao, Bencheng and Chen, Jiajie and Zhou, Helong and Zhang, Qian and Liu, Wenyu and Huang, Chang and Wang, Xinggang},
  booktitle={ICCV},
  year={2023}
}
```