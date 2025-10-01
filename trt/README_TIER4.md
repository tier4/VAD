# TensorRT Deployment for VAD with CARLA Tier4 Format

## Overview

This documentation provides comprehensive instructions for deploying VAD models with TensorRT, specifically supporting the CARLA Tier4 coordinate system and custom configurations. The implementation now dynamically handles different model variants including nuScenes, CARLA/B2D, and Tier4 formats.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Configuration System](#configuration-system)
4. [Tier4 Coordinate System](#tier4-coordinate-system)
5. [Export Process](#export-process)
6. [Inference Process](#inference-process)
7. [Engine Management](#engine-management)
8. [Command Reference](#command-reference)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### Environment Setup

```bash
# Ensure CUDA environment is configured
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate VAD environment
conda activate vad  # or mamba activate vad

# Verify TensorRT installation
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

### Required Files

1. **Checkpoint Files** (relative to VAD root):
   - nuScenes: `ckpts/VAD_tiny.pth`, `ckpts/[base_checkpoint_not_available]`
   - CARLA Tier4: `ckpts/carla_tier4_epoch_20.pth`, `ckpts/[base_checkpoint_not_available]`

2. **Configuration Files** (relative to VAD root):
   - Standard: `configs/VAD/VAD_tiny_e2e.py`, `configs/VAD/VAD_base_e2e.py`
   - Tier4: `configs/VAD/VAD_tiny_carla_tier4.py`, `configs/VAD/VAD_base_carla_tier4.py`

## Architecture Overview

The TRT deployment consists of three main components:

### 1. Model Export Scripts
- `export_no_prev.py`: Exports the model for first frame (without previous BEV)
- `export_prev.py`: Exports the model for subsequent frames (with previous BEV)
- `save_data.py`: Saves intermediate data for validation

### 2. Configuration Utilities
- `config_utils.py`: Dynamically extracts TRT-relevant parameters from VAD configs
- Supports multiple coordinate systems and model variants
- Automatically detects Tier4 configurations

### 3. TensorRT Inference
- `test_tensorrt.py`: Main inference script with dynamic configuration support
- Engine files organized by model variant
- Support for FP16 optimization

## Configuration System

### Dynamic Configuration Loading

The system now automatically extracts configuration parameters from VAD config files:

```python
from config_utils import (
    get_bev_dimensions,      # Extract BEV height and width
    get_grid_length,         # Calculate grid length from voxel sizes
    get_class_counts,        # Get object and map class counts
    get_coordinate_system,   # Detect coordinate system (tier4/standard)
    get_model_variant_name,  # Generate model variant identifier
    get_engine_paths,        # Get engine file paths
    print_config_summary     # Print configuration summary
)
```

### Supported Configurations

| Configuration | BEV Size | Voxel Size | Object Classes | Map Classes | Coordinate System |
|--------------|----------|------------|----------------|-------------|-------------------|
| nuScenes Tiny | 117×58 | [0.512, 0.512, 4] | 10 | 3 | Standard |
| nuScenes Base | 200×200 | [0.3, 0.3, 4] | 10 | 3 | Standard |
| CARLA Tier4 Tiny | 64×120 | [0.5, 0.5, 4] | 9 | 6 | Tier4 |
| CARLA Tier4 Base | 64×120 | [0.5, 0.5, 4] | 9 | 6 | Tier4 |

### Configuration Parameters

#### Point Cloud Range
- **Standard nuScenes**: `[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]`
- **Tier4 CARLA**: `[-30.0, -16.0, -0.16, 30.0, 16.0, 3.84]`

#### Object Classes
- **nuScenes** (10 classes): car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone
- **CARLA Tier4** (9 classes): car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian

#### Map Classes
- **nuScenes** (3 classes): divider, ped_crossing, boundary
- **CARLA Tier4** (6 classes): road_divider, lane_divider, ped_crossing, contours, centerline, others

## Tier4 Coordinate System

The Tier4 coordinate system uses a specific transformation from the standard coordinate system:

### Coordinate Transformation
```
Y -> X  (new X-axis is old Y-axis)
-X -> Y (new Y-axis is negative old X-axis)
Z -> Z  (Z-axis unchanged)
```

### Implementation in C++ (coordinate_config.hpp)
```cpp
// Axis rotation matrix for Tier4
static constexpr float axis_rotation[9] = {
    0.0f, 1.0f, 0.0f,   // new X = old Y
    -1.0f, 0.0f, 0.0f,  // new Y = old -X
    0.0f, 0.0f, 1.0f    // new Z = old Z
};
```

### CAN Bus Data Transformation
When using Tier4 coordinates, CAN bus data must be transformed:
- Velocity components are rotated
- Acceleration components are rotated
- Yaw angle is adjusted by π/2
- BEV rotation angle is adjusted

## Export Process

### Step 1: Prepare Environment

```bash
# Navigate to export directory
cd /mnt/sda1/VAD/trt/export_eval

# Verify configuration utility works
python -c "from config_utils import print_config_summary; print('Config utils loaded successfully')"
```

### Step 2: Export ONNX Models

The export process creates separate ONNX models for different components of the VAD architecture.

#### For nuScenes Standard Model:

##### Export First Frame Model (No Previous BEV):
```bash
python export_no_prev.py ../../configs/VAD/VAD_tiny_e2e.py ../../ckpts/VAD_tiny.pth --launcher none --out results_no_prev.pkl --eval bbox --tmpdir tmp_no_prev
```

##### Export Subsequent Frames Model (With Previous BEV):
```bash
python export_prev.py ../../configs/VAD/VAD_tiny_e2e.py ../../ckpts/VAD_tiny.pth --launcher none --out results_prev.pkl --eval bbox --tmpdir tmp_prev
```

#### For CARLA Tier4 Model:

##### Export First Frame Model:
```bash
python export_no_prev.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --out results_tier4_no_prev.pkl --eval bbox --tmpdir tmp_tier4_no_prev --deterministic
```

##### Export Subsequent Frames Model:
```bash
python export_prev.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --out results_tier4_prev.pkl --eval bbox --tmpdir tmp_tier4_prev --deterministic
```

#### For Base Models (Larger Architecture):

##### nuScenes Base Model:
```bash
# First frame
python export_no_prev.py ../../configs/VAD/VAD_base_e2e.py ../../ckpts/[base_checkpoint_not_available] --launcher none --out results_base_no_prev.pkl --eval bbox --cfg-options model.img_backbone.with_cp=False

# Subsequent frames
python export_prev.py ../../configs/VAD/VAD_base_e2e.py ../../ckpts/[base_checkpoint_not_available] --launcher none --out results_base_prev.pkl --eval bbox --cfg-options model.img_backbone.with_cp=False
```

##### CARLA Tier4 Base Model:
```bash
# First frame
python export_no_prev.py ../../configs/VAD/VAD_base_carla_tier4.py ../../ckpts/[base_checkpoint_not_available] --launcher none --out results_tier4_base_no_prev.pkl --eval bbox --cfg-options model.img_backbone.with_cp=False

# Subsequent frames
python export_prev.py ../../configs/VAD/VAD_base_carla_tier4.py ../../ckpts/[base_checkpoint_not_available] --launcher none --out results_tier4_base_prev.pkl --eval bbox --cfg-options model.img_backbone.with_cp=False
```

### Step 3: Verify ONNX Export

After export, verify that ONNX files are created:

```bash
# List generated ONNX files
find scratch/ -name "*.onnx" -type f -exec ls -lh {} \;

# Expected output structure:
# scratch/carla_tier4_tiny/vadv1.extract_img_feat/sim_vadv1.extract_img_feat.onnx
# scratch/carla_tier4_tiny/vadv1.pts_bbox_head.forward/sim_vadv1.pts_bbox_head.forward.onnx
# scratch/carla_tier4_tiny/vadv1_prev.pts_bbox_head.forward/sim_vadv1_prev.pts_bbox_head.forward.onnx
```

### Step 4: Convert ONNX to TensorRT Engines

The conversion happens automatically during export, but can also be done manually:

```bash
# Manual conversion using trtexec (if needed)
trtexec --onnx=scratch/carla_tier4_tiny/vadv1.extract_img_feat/sim_vadv1.extract_img_feat.onnx --saveEngine=scratch/carla_tier4_tiny/vadv1.extract_img_feat/sim_vadv1.extract_img_feat_fp16.engine --fp16 --workspace=4096 --verbose --buildOnly
```

### Step 5: Validate Export Results

Check the export results pickle file:

```bash
# Validate export results
python -c "
import pickle
with open('results_tier4_no_prev.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f'Number of samples: {len(data[\"results\"])}')
    sample = list(data['results'].keys())[0]
    print(f'First sample token: {sample}')
    print(f'Number of detections: {len(data[\"results\"][sample])}')
"
```

### Engine File Organization

Engines are organized by model variant:
```
scratch/
├── nuscenes_tiny/
│   ├── vadv1.extract_img_feat/
│   │   └── sim_vadv1.extract_img_feat_fp16.engine
│   ├── vadv1.pts_bbox_head.forward/
│   │   └── sim_vadv1.pts_bbox_head.forward.engine
│   └── vadv1_prev.pts_bbox_head.forward/
│       └── sim_vadv1_prev.pts_bbox_head.forward.engine
└── carla_tier4_tiny/
    ├── vadv1.extract_img_feat/
    │   └── sim_vadv1.extract_img_feat_fp16.engine
    ├── vadv1.pts_bbox_head.forward/
    │   └── sim_vadv1.pts_bbox_head.forward.engine
    └── vadv1_prev.pts_bbox_head.forward/
        └── sim_vadv1_prev.pts_bbox_head.forward.engine
```

## Inference Process

### Step 1: Prepare for TensorRT Inference

```bash
# Navigate to export directory
cd /mnt/sda1/VAD/trt/export_eval

# Verify engines exist
ls -la scratch/*/vadv1*/

# Set TensorRT logging level (optional)
export TRT_LOGGER_LEVEL=INFO  # Options: VERBOSE, INFO, WARNING, ERROR
```

### Step 2: Run TensorRT Inference

#### Basic Inference Commands:

##### For nuScenes Tiny Model:
```bash
python test_tensorrt.py ../../configs/VAD/VAD_tiny_e2e.py ../../ckpts/VAD_tiny.pth --launcher none --eval bbox --out trt_results_nuscenes.pkl --tmpdir tmp_trt_nuscenes --gpu-collect
```

##### For CARLA Tier4 Tiny Model:
```bash
python test_tensorrt.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --eval bbox --out trt_results_tier4.pkl --tmpdir tmp_trt_tier4 --deterministic --seed 42
```

#### Advanced Inference Options:

##### With Custom Evaluation Options:
```bash
python test_tensorrt.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --eval bbox --eval-options classwise=True jsonfile_prefix=./results_tier4 metric=bbox --out trt_results_detailed.pkl
```

##### With Configuration Overrides:
```bash
python test_tensorrt.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --eval bbox --cfg-options data.test.samples_per_gpu=1 data.workers_per_gpu=4 --out trt_results_custom.pkl
```

##### For Base Models:
```bash
# nuScenes Base Model
python test_tensorrt.py ../../configs/VAD/VAD_base_e2e.py ../../ckpts/[base_checkpoint_not_available] --launcher none --eval bbox --out trt_results_base.pkl --cfg-options model.img_backbone.with_cp=False data.test.samples_per_gpu=1

# CARLA Tier4 Base Model
python test_tensorrt.py ../../configs/VAD/VAD_base_carla_tier4.py ../../ckpts/[base_checkpoint_not_available] --launcher none --eval bbox --out trt_results_tier4_base.pkl --cfg-options model.img_backbone.with_cp=False data.test.samples_per_gpu=1
```

### Step 3: Monitor Inference Progress

The inference will display:

```
============================================================
TRT Configuration Summary
============================================================
Model Variant: carla_tier4_tiny
Coordinate System: tier4
BEV Dimensions: 64 x 120
Voxel Sizes: [0.5, 0.5, 4]
Grid Length: [0.5, 0.5]
Object Classes: 9
Map Classes: 6
Point Cloud Range: [-30.0, -16.0, -0.16, 30.0, 16.0, 3.84]
============================================================
Engine Paths:
  Image Feature Extractor: scratch/carla_tier4_tiny/vadv1.extract_img_feat/sim_vadv1.extract_img_feat_fp16.engine
  BBox Head (no prev): scratch/carla_tier4_tiny/vadv1.pts_bbox_head.forward/sim_vadv1.pts_bbox_head.forward.engine
  BBox Head (with prev): scratch/carla_tier4_tiny/vadv1_prev.pts_bbox_head.forward/sim_vadv1_prev.pts_bbox_head.forward.engine
============================================================
Loading TensorRT engines...
[TRT] Loading engine: vadv1.extract_img_feat
[TRT] Loading engine: vadv1.pts_bbox_head.forward
[TRT] Loading engine: vadv1_prev.pts_bbox_head.forward
Engines loaded successfully!

Processing samples: [######################] 100%
```

### Step 4: Analyze Results

#### View Inference Results:
```bash
# Check output pickle file
python -c "
import pickle
import numpy as np

with open('trt_results_tier4.pkl', 'rb') as f:
    data = pickle.load(f)
    
print('=== TensorRT Inference Results ===')
print(f'Total samples: {len(data[\"results\"])}')

# Analyze first sample
sample_token = list(data['results'].keys())[0]
detections = data['results'][sample_token]
print(f'\\nSample {sample_token}:')
print(f'  Detections: {len(detections)}')

if detections:
    det = detections[0]
    print(f'  Detection keys: {list(det.keys())}')
    if 'fut_traj' in det:
        traj = np.array(det['fut_traj'])
        print(f'  Trajectory shape: {traj.shape}')
        print(f'  Trajectory range: [{traj.min():.2f}, {traj.max():.2f}]')
"
```

#### Compare with PyTorch Results:
```bash
# Run PyTorch inference for comparison
python ../../tools/test.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --eval bbox --out pytorch_results.pkl

# Compare results
python -c "
import pickle
import numpy as np

with open('trt_results_tier4.pkl', 'rb') as f:
    trt_data = pickle.load(f)
    
with open('pytorch_results.pkl', 'rb') as f:
    pytorch_data = pickle.load(f)

print('=== Comparison: TensorRT vs PyTorch ===')
print(f'TRT samples: {len(trt_data[\"results\"])}')
print(f'PyTorch samples: {len(pytorch_data[\"results\"])}')

# Compare detection counts
sample = list(trt_data['results'].keys())[0]
trt_dets = len(trt_data['results'][sample])
pytorch_dets = len(pytorch_data['results'][sample])
print(f'\\nSample {sample}:')
print(f'  TRT detections: {trt_dets}')
print(f'  PyTorch detections: {pytorch_dets}')
"
```

### Step 5: Performance Profiling

#### Profile TensorRT Inference:
```bash
# Run with profiling enabled
python test_tensorrt.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --eval bbox --out trt_profile.pkl --cfg-options profiling=True profile_samples=100
```

#### Measure Inference Speed:
```bash
# Benchmark inference speed
python -c "
import time
import torch
from test_tensorrt import main

# Warm up
main(['../../configs/VAD/VAD_tiny_carla_tier4.py', 
      '../../ckpts/carla_tier4_epoch_20.pth',
      '--launcher', 'none', '--eval', 'bbox',
      '--out', 'warmup.pkl'])

# Measure
start = time.time()
main(['../../configs/VAD/VAD_tiny_carla_tier4.py',
      '../../ckpts/carla_tier4_epoch_20.pth', 
      '--launcher', 'none', '--eval', 'bbox',
      '--out', 'benchmark.pkl'])
end = time.time()

print(f'Total inference time: {end-start:.2f} seconds')
"
```

### Dynamic Tensor Shapes

The system automatically adjusts tensor shapes based on configuration:

```python
# BEV embed size calculated dynamically
bev_embed_size = bev_h * bev_w  # 64×120=7680 for Tier4, 117×58=6786 for nuScenes

# Output tensor shapes adjusted per configuration
# For Tier4:
lut_tier4 = [
    ("bev_embed", [7680, 1, 256]),         # 64×120 BEV
    ("all_cls_scores", [3, 1, 300, 9]),    # 9 object classes
    ("all_bbox_preds", [3, 1, 300, 10]),
    ("all_traj_preds", [3, 1, 300, 6, 12]),
    ("all_traj_cls_scores", [3, 1, 300, 6]),
    ("map_all_cls_scores", [3, 1, 100, 6]), # 6 map classes
    ("map_all_bbox_preds", [3, 1, 100, 4]),
    ("map_all_pts_preds", [3, 1, 100, 20, 2]),
    ("ego_fut_preds", [1, 3, 6, 2])
]

# For nuScenes:
lut_nuscenes = [
    ("bev_embed", [6786, 1, 256]),         # 117×58 BEV
    ("all_cls_scores", [3, 1, 300, 10]),   # 10 object classes
    ("all_bbox_preds", [3, 1, 300, 10]),
    ("all_traj_preds", [3, 1, 300, 6, 12]),
    ("all_traj_cls_scores", [3, 1, 300, 6]),
    ("map_all_cls_scores", [3, 1, 100, 3]), # 3 map classes
    ("map_all_bbox_preds", [3, 1, 100, 4]),
    ("map_all_pts_preds", [3, 1, 100, 20, 2]),
    ("ego_fut_preds", [1, 3, 6, 2])
]
```

## Engine Management

### Automatic Engine Path Resolution

The system automatically determines engine paths based on the configuration:

```python
from config_utils import get_engine_paths

# Automatically generates paths like:
# scratch/carla_tier4_tiny/vadv1.extract_img_feat/sim_vadv1.extract_img_feat_fp16.engine
engine_paths = get_engine_paths(cfg)
```

### Multi-Configuration Support

To support multiple configurations simultaneously:

1. **Separate Engine Directories**: Each model variant has its own engine directory
2. **Configuration Detection**: Automatically detects configuration from config file
3. **Dynamic Loading**: Loads appropriate engine based on detected configuration

### CUDA Version Compatibility

For supporting both CUDA 11.8 and 12.8:

```python
import torch

# Detect CUDA version at runtime
cuda_version = torch.version.cuda.replace('.', '')
engine_base_path = f"scratch/cuda{cuda_version}/{model_variant}/"
```

## Command Reference

### Complete Export Workflow

#### 1. Full Export Pipeline for nuScenes:
```bash
# Set up environment
cd /mnt/sda1/VAD/trt/export_eval
export CUDA_VISIBLE_DEVICES=0

# Export first frame model
python export_no_prev.py ../../configs/VAD/VAD_tiny_e2e.py ../../ckpts/VAD_tiny.pth --launcher none --out export_nuscenes_no_prev.pkl --eval bbox --tmpdir tmp_export_no_prev --deterministic --seed 0

# Export subsequent frames model  
python export_prev.py ../../configs/VAD/VAD_tiny_e2e.py ../../ckpts/VAD_tiny.pth --launcher none --out export_nuscenes_prev.pkl --eval bbox --tmpdir tmp_export_prev --deterministic --seed 0

# Verify ONNX files
find scratch/nuscenes_tiny -name "*.onnx" -exec ls -lh {} \;

# Convert to engines if needed
for onnx_file in scratch/nuscenes_tiny/*/*.onnx; do
    engine_file="${onnx_file%.onnx}_fp16.engine"
    if [ ! -f "$engine_file" ]; then
        echo "Building engine for $onnx_file"
        trtexec --onnx=$onnx_file --saveEngine=$engine_file --fp16
    fi
done
```

#### 2. Full Export Pipeline for CARLA Tier4:
```bash
# Set up environment
cd /mnt/sda1/VAD/trt/export_eval
export CUDA_VISIBLE_DEVICES=0

# Export first frame model with verbose output
python export_no_prev.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --out export_tier4_no_prev.pkl --eval bbox --tmpdir tmp_tier4_no_prev --deterministic --seed 0 --cfg-options log_level=DEBUG

# Export subsequent frames model
python export_prev.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --out export_tier4_prev.pkl --eval bbox --tmpdir tmp_tier4_prev --deterministic --seed 0

# Verify export success
echo "Checking exported files..."
ls -la scratch/carla_tier4_tiny/vadv1.extract_img_feat/
ls -la scratch/carla_tier4_tiny/vadv1.pts_bbox_head.forward/
ls -la scratch/carla_tier4_tiny/vadv1_prev.pts_bbox_head.forward/
```

### Complete Inference Workflow

#### 1. Basic Inference with Metrics:
```bash
# nuScenes inference with full evaluation
python test_tensorrt.py ../../configs/VAD/VAD_tiny_e2e.py ../../ckpts/VAD_tiny.pth --launcher none --eval bbox --out results_nuscenes.pkl --eval-options classwise=True jsonfile_prefix=./nuscenes_metrics metric=bbox

# CARLA Tier4 inference with full evaluation  
python test_tensorrt.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --eval bbox --out results_tier4.pkl --eval-options classwise=True jsonfile_prefix=./tier4_metrics metric=bbox
```

#### 2. Batch Processing Multiple Checkpoints:
```bash
#!/bin/bash
# batch_inference.sh

CONFIGS=(
    "../../configs/VAD/VAD_tiny_e2e.py"
    "../../configs/VAD/VAD_tiny_carla_tier4.py"
)

CHECKPOINTS=(
    "../../ckpts/VAD_tiny.pth"
    "../../ckpts/carla_tier4_epoch_20.pth"
)

OUTPUT_NAMES=(
    "nuscenes_tiny"
    "carla_tier4_tiny"
)

for i in ${!CONFIGS[@]}; do
    echo "Processing ${OUTPUT_NAMES[$i]}..."
    python test_tensorrt.py ${CONFIGS[$i]} ${CHECKPOINTS[$i]} --launcher none --eval bbox --out results_${OUTPUT_NAMES[$i]}.pkl --tmpdir tmp_${OUTPUT_NAMES[$i]}
done
```

### Utility Commands

#### 1. Configuration Verification:
```bash
# Test configuration extraction
python -c "
from config_utils import *
from mmcv import Config

# Test nuScenes config
cfg = Config.fromfile('../../configs/VAD/VAD_tiny_e2e.py')
print('=== nuScenes Configuration ===')
print_config_summary(cfg)

# Test Tier4 config
cfg = Config.fromfile('../../configs/VAD/VAD_tiny_carla_tier4.py')
print('\\n=== CARLA Tier4 Configuration ===')
print_config_summary(cfg)
"
```

#### 2. Engine Information:
```bash
# Inspect engine details
python -c "
from bev_deploy.trt.inference import InferTrt

engine_path = 'scratch/carla_tier4_tiny/vadv1.extract_img_feat/sim_vadv1.extract_img_feat_fp16.engine'
infer = InferTrt()
infer.read(engine_path)

print('=== Engine Information ===')
print(f'Engine: {engine_path}')
print(f'Input tensors:')
for name, shape in infer.input_shapes.items():
    print(f'  {name}: {shape}')
print(f'Output tensors:')
for name, shape in infer.output_shapes.items():
    print(f'  {name}: {shape}')
"
```

#### 3. Data Validation:
```bash
# Validate saved data for TensorRT
python save_data.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --out saved_data.pkl --save-tensor-data

# Check saved data
python -c "
import pickle
with open('saved_data.pkl', 'rb') as f:
    data = pickle.load(f)
    print('Saved data keys:', list(data.keys()))
    if 'tensor_data' in data:
        print('Tensor data shapes:')
        for k, v in data['tensor_data'].items():
            print(f'  {k}: {v.shape if hasattr(v, \"shape\") else type(v)}')
"
```

#### 4. Clean Build:
```bash
# Clean and rebuild engines
#!/bin/bash

# Remove existing engines
rm -rf scratch/*/vadv1*/sim*.engine

# Rebuild for nuScenes
python export_no_prev.py ../../configs/VAD/VAD_tiny_e2e.py ../../ckpts/VAD_tiny.pth --launcher none --out rebuild_nuscenes_no_prev.pkl --eval bbox

python export_prev.py ../../configs/VAD/VAD_tiny_e2e.py ../../ckpts/VAD_tiny.pth --launcher none --out rebuild_nuscenes_prev.pkl --eval bbox

# Rebuild for Tier4
python export_no_prev.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --out rebuild_tier4_no_prev.pkl --eval bbox

python export_prev.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --out rebuild_tier4_prev.pkl --eval bbox
```

### Advanced Usage

#### 1. Multi-GPU Export (Distributed):
```bash
# Export with multiple GPUs
python -m torch.distributed.run --nproc_per_node=2 --master_port=29500 export_no_prev.py ../../configs/VAD/VAD_base_carla_tier4.py ../../ckpts/[base_checkpoint_not_available] --launcher pytorch --out distributed_export.pkl --eval bbox
```

#### 2. Memory-Optimized Export:
```bash
# Export large models with memory optimization
python export_no_prev.py ../../configs/VAD/VAD_base_carla_tier4.py ../../ckpts/[base_checkpoint_not_available] --launcher none --out memory_opt_export.pkl --eval bbox --cfg-options model.img_backbone.with_cp=False model.img_backbone.frozen_stages=3 data.test.samples_per_gpu=1
```

#### 3. Custom Dataset Testing:
```bash
# Test with custom dataset configuration
python test_tensorrt.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --eval bbox --out custom_dataset_results.pkl --cfg-options data.test.type=CustomDataset data.test.data_root=/path/to/custom/data data.test.ann_file=/path/to/annotations.json
```

#### 4. Debugging Failed Exports:
```bash
# Debug export with verbose logging
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TRT_LOGGER_LEVEL=VERBOSE

python export_no_prev.py ../../configs/VAD/VAD_tiny_carla_tier4.py ../../ckpts/carla_tier4_epoch_20.pth --launcher none --out debug_export.pkl --eval bbox --cfg-options log_level=DEBUG 2>&1 | tee export_debug.log

# Analyze log for errors
grep -E "ERROR|FAIL|Exception" export_debug.log
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Engine File Not Found
**Error**: `FileNotFoundError: Engine file not found`
**Solution**: 
- Ensure you've exported the ONNX models first
- Check that engine paths match your configuration
- Verify the model variant name matches expected format

#### 2. Shape Mismatch Errors
**Error**: `RuntimeError: Expected shape [100, 100] but got [64, 120]`
**Solution**:
- Ensure you're using the correct config file
- Regenerate engines after modifying configurations
- Check that BEV dimensions are correctly extracted

#### 3. Class Count Mismatch
**Error**: `IndexError: index 9 is out of bounds for dimension with size 9`
**Solution**:
- Verify the number of classes in your config
- Ensure engine was built with correct class count
- Check that dataset classes match model configuration

#### 4. Coordinate System Issues
**Error**: Incorrect object positions or trajectories
**Solution**:
- Verify coordinate system detection (tier4 vs standard)
- Check CAN bus data transformation
- Ensure point cloud range matches expected values

### Validation Steps

1. **Verify Configuration Loading**:
```bash
cd /mnt/sda1/VAD/trt/export_eval
python test_config_utils.py
```

2. **Check Engine Compatibility**:
```python
# Verify engine input/output shapes
from bev_deploy.trt.inference import InferTrt
infer = InferTrt()
infer.read("path/to/engine")
print(infer)  # Shows input/output tensor information
```

3. **Compare with PyTorch Output**:
- Run inference with PyTorch model
- Run inference with TensorRT engine
- Compare outputs to ensure consistency

### Performance Optimization

1. **FP16 Optimization**: Engines use FP16 by default for better performance
2. **Batch Size**: Currently set to 1, can be increased for batch processing
3. **Memory Management**: Ensure sufficient GPU memory for larger models

### Debug Mode

To enable detailed debugging:
```python
# In test_tensorrt.py, add verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print intermediate tensor shapes
print(f"BEV features shape: {mlvl_feats[0].shape}")
print(f"Lidar2img shape: {lidar2img.shape}")
print(f"CAN bus data: {m[0]['can_bus']}")
```

## Advanced Features

### Custom Configuration Support

To add support for a new configuration:

1. **Update config_utils.py**:
- Add detection logic in `get_coordinate_system()`
- Update `get_model_variant_name()` for new variant
- Adjust class counts if needed

2. **Create Engine Directory**:
```bash
mkdir -p scratch/your_variant_name/
```

3. **Export and Convert Models**:
Follow the standard export process with your config file

### Integration with CI/CD

For automated testing:
```bash
#!/bin/bash
# test_all_configs.sh

configs=(
    "../../configs/VAD/VAD_tiny_e2e.py"
    "../../configs/VAD/VAD_tiny_carla_tier4.py"
    "../../configs/VAD/VAD_base_e2e.py"
)

for config in "${configs[@]}"; do
    echo "Testing $config"
    python test_tensorrt.py $config checkpoint.pth --launcher none --eval bbox
done
```

## References

- [VAD Paper](https://arxiv.org/abs/2303.12077)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [CARLA B2D Documentation](https://github.com/rethinklab/Bench2DriveZoo)
- [nuScenes Dataset](https://www.nuscenes.org/)

## Contributors

This TensorRT implementation with Tier4 support was developed to enable efficient deployment of VAD models across different coordinate systems and configurations.