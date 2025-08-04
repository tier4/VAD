# VAD Coordinate System Transformation - Merged Documentation

## Table of Contents
1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Coordinate System Definitions](#coordinate-system-definitions)
4. [Data Flow Visualization](#data-flow-visualization)
5. [Transformation Implementation Details](#transformation-implementation-details)
6. [Bug Fixes Applied](#bug-fixes-applied)
7. [Training Pipeline](#training-pipeline)
8. [Inference and Output](#inference-and-output)
9. [Evaluation Pipeline](#evaluation-pipeline)
10. [TensorRT Deployment](#tensorrt-deployment)
11. [BEV Grid Considerations](#bev-grid-considerations)
12. [Benefits](#benefits)
13. [Future Improvements](#future-improvements)
14. [Complete Transformation Checklist](#complete-transformation-checklist)
15. [Key Insights](#key-insights)


## Overview

The VAD model transforms data from the original nuScenes LiDAR coordinate system to a new ego-centered coordinate system. This document details every coordinate transformation throughout the codebase. The transformation changes the model from using a LiDAR-centered coordinate system to an ego-centered coordinate system with axis rotation, improving compatibility with standard autonomous driving frameworks.

## Motivation

The original VAD implementation uses a LiDAR-centered coordinate system, which places the origin at the LiDAR sensor position (typically ~1.5m above the vehicle center). Many autonomous driving systems prefer ego-centered coordinates with the origin at the vehicle's rear axle. Additionally, different coordinate conventions exist across frameworks regarding axis orientations.

## Coordinate System Definitions

### Original LiDAR Coordinate System
```
      ^ X (Forward)
      |
      |
Y <---+  Z (Up)
(Left)
```
- **Origin**: LiDAR sensor position
- **X-axis**: Forward (positive forward)
- **Y-axis**: Left (positive left)
- **Z-axis**: Up (positive up)
- **Range**: X: [-15, 15]m, Y: [-30, 30]m, Z: [-2, 2]m

### New Ego Coordinate System
```
      ^ Y (Backward)
      |
      |
X <---+  Z (Up)
(Left)
```
- **Origin**: Ego vehicle center (rear axle)
- **X-axis**: Left (old Y-axis)
- **Y-axis**: Backward (old -X-axis)
- **Z-axis**: Up (same as before)
- **Range**: X: [-30, 30]m, Y: [-16, 16]m, Z: [-0.16, 3.84]m

### Transformation Matrix
The transformation consists of:
1. Translation from LiDAR to ego using calibration data
2. Rotation from LiDAR to ego using calibration data
3. Axis rotation: Y→X, -X→Y, Z→Z

Axis rotation matrix:
```python
# Axis rotation matrix: Y→X, -X→Y, Z→Z
axis_rotation = [[0,  1, 0],
                 [-1, 0, 0],
                 [0,  0, 1]]
```
This represents a 90° clockwise rotation when viewed from above.

## Data Flow Visualization

```
┌─────────────────────┐
│   nuScenes Data     │
│  (LiDAR Coords)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Data Converter     │ ← Generates dataset with original coordinates
│(vad_nuscenes_conv.) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Dataset Loading   │ ← Transforms to new ego coordinates
│(nuscenes_vad_dataset)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Model Training    │ ← All operations in new coordinates
│   (VAD modules)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Model Output      │ ← Predictions in new coordinates
│  (Detection Head)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Evaluation       │ ← Transforms back to original for metrics
│(output_to_nusc_box) │
└─────────────────────┘
```

## Transformation Implementation Details

### 1. Data Loading Stage (nuscenes_vad_dataset.py)

#### 1.1 Transformation Helper Functions

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 986-988
@staticmethod
def get_axis_rotation_matrix():
    '''Get the axis rotation matrix: Y->X, -X->Y, Z->Z'''
    return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
```

#### 1.2 Point Transformation

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 991-1022
@staticmethod
def transform_lidar_to_ego_new(points, lidar2ego_translation, lidar2ego_rotation):
    '''Transform points from LiDAR coordinates to ego coordinates with axis rotation.'''
    # First: LiDAR to ego
    lidar2ego_rot_mat = Quaternion(lidar2ego_rotation).rotation_matrix
    points_ego = points_3d @ lidar2ego_rot_mat.T + lidar2ego_translation

    # Second: Apply axis rotation
    axis_rotation = VADCustomNuScenesDataset.get_axis_rotation_matrix()
    points_new = points_ego @ axis_rotation.T

    return points_new
```

#### 1.3 Velocity Transformation

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 1025-1053
@staticmethod
def transform_velocity_to_ego_new(velocity, lidar2ego_rotation):
    '''Transform velocity from LiDAR to new ego coordinate system.'''
    # Apply LiDAR to ego rotation
    lidar2ego_rot_mat = Quaternion(lidar2ego_rotation).rotation_matrix
    velocity_ego = velocity @ lidar2ego_rot_mat.T

    # Apply axis rotation: Y->X, -X->Y
    axis_rotation = VADCustomNuScenesDataset.get_axis_rotation_matrix()
    velocity_new = velocity_ego @ axis_rotation.T

    return velocity_new
```

#### 1.4 Yaw Angle Transformation

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 1055-1076
@staticmethod
def transform_yaw_to_ego_new(yaw, lidar2ego_rotation):
    '''Transform yaw angle from LiDAR to new ego coordinate system.'''
    # Apply LiDAR to ego rotation
    lidar2ego_yaw = quaternion_yaw(Quaternion(lidar2ego_rotation))
    yaw_ego = yaw + lidar2ego_yaw

    # Apply 90-degree rotation due to axis change (Y->X, -X->Y)
    # Coordinate system rotates clockwise by 90°, so yaw must decrease by 90°
    yaw_new = yaw_ego - np.pi / 2

    # Normalize to [-pi, pi]
    yaw_new = np.arctan2(np.sin(yaw_new), np.cos(yaw_new))

    return yaw_new
```

### 2. Annotation Transformation (get_ann_info)

#### 2.1 Bounding Box Transformation

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 1396-1418
# Transform box centers
box_centers = gt_bboxes_3d[:, :3]
box_centers_new = self.transform_lidar_to_ego_new(
    box_centers, lidar2ego_translation, lidar2ego_rotation
)

# Transform box dimensions (w, l, h) - swap w and l due to axis rotation
box_dims = gt_bboxes_3d[:, 3:6].copy()
box_dims_new = box_dims.copy()
box_dims_new[:, 0] = box_dims[:, 1]  # new w = old l
box_dims_new[:, 1] = box_dims[:, 0]  # new l = old w

# Transform yaw angles
box_yaw = gt_bboxes_3d[:, 6]
box_yaw_new = self.transform_yaw_to_ego_new(box_yaw, lidar2ego_rotation)

# Reconstruct bounding boxes
gt_bboxes_3d_new = np.concatenate([
    box_centers_new,
    box_dims_new,
    box_yaw_new[:, None]
], axis=1)
```

#### 2.2 Future Trajectory Transformation

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 1438-1444
# Transform future trajectories to new coordinate system
if len(gt_fut_trajs) > 0:
    fut_trajs_shape = gt_fut_trajs.shape
    gt_fut_trajs_flat = gt_fut_trajs.reshape(-1, 2)
    gt_fut_trajs_new = self.transform_lidar_to_ego_new(
        gt_fut_trajs_flat, lidar2ego_translation, lidar2ego_rotation
    )
    gt_fut_trajs_new = gt_fut_trajs_new.reshape(fut_trajs_shape)
```

#### 2.3 Goal Direction Transformation

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 1447-1454
# Transform future goals - these are direction classes (0-8), not coordinates
# Each class represents 45 degrees, and we're rotating by 90 degrees (π/2)
# So we add 2 classes (90/45 = 2) and wrap around
gt_fut_goal_new = gt_fut_goal.copy()
non_static_mask = gt_fut_goal < 9
gt_fut_goal_new[non_static_mask] = (gt_fut_goal[non_static_mask] + 2) % 8
```

### 3. Camera Data Transformation (get_data_info)

#### 3.1 CAN Bus Data Transformation

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 1640-1661
# Transform velocity
vel_x, vel_y, vel_z = can_bus[10:13]
can_bus[10] = vel_y    # Y -> X
can_bus[11] = -vel_x   # -X -> Y

# Transform acceleration
acc_x, acc_y, acc_z = can_bus[13:16]
can_bus[13] = acc_y    # Y -> X
can_bus[14] = -acc_x   # -X -> Y

# Adjust yaw angles for coordinate system rotation (90 degrees)
# can_bus[16] is in radians, can_bus[17] is in degrees
can_bus[16] -= np.pi / 2   # Subtract 90 degrees in radians
can_bus[17] -= 90          # Subtract 90 degrees in degrees
# Normalize degrees to [0, 360)
if can_bus[17] < 0:
    can_bus[17] += 360
```

#### 3.2 Camera Transformation Matrix

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 1080-1105
@staticmethod
def get_ego_new_to_cam_matrix(lidar2cam_rt, lidar2ego_translation, lidar2ego_rotation):
    '''Get transformation matrix from new ego coordinate system to camera.'''
    # Build transformation matrices
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(lidar2ego_rotation).rotation_matrix
    lidar2ego[:3, 3] = lidar2ego_translation

    # Axis rotation matrix (4x4)
    axis_rotation_4x4 = np.eye(4)
    axis_rotation_4x4[:3, :3] = VADCustomNuScenesDataset.get_axis_rotation_matrix()

    # ego_new to lidar: inverse of (lidar->ego->axis_rotation)
    ego_new_to_lidar = np.linalg.inv(axis_rotation_4x4 @ lidar2ego)

    # ego_new to cam: ego_new->lidar->cam
    ego_new_to_cam = lidar2cam_rt @ ego_new_to_lidar

    return ego_new_to_cam
```

### 4. Map Data Transformation

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 1235-1243
# Transform map elements to new ego coordinate system
for instance in anns:
    if isinstance(instance, LineString):
        # Get coordinates from LineString
        coords = np.array(instance.coords)
        # Transform to new ego coordinate system
        coords_new = self.transform_lidar_to_ego_new(
            coords, lidar2ego_translation, lidar2ego_rotation
        )
        # Create new LineString with transformed coordinates
        new_instance = LineString(coords_new)
```

### Configuration Updates

#### Accurate Range Calculation
The coordinate ranges are calculated based on actual nuScenes sensor calibration data:

**LiDAR Mounting Statistics** (from 34,149 samples in v1.0-trainval):
- **Mean position**: X=0.9652m forward, Y=0.0000m, Z=1.8403m above rear axle
- **Three vehicle configurations**:
  - Vehicle 1: [0.8911, 0.0000, 1.8429]
  - Vehicle 2: [0.9437, 0.0000, 1.8402]
  - Vehicle 3: [0.9858, 0.0000, 1.8402]
- **Standard deviation**: X=0.023m, Y=0.000m, Z=0.0004m

Using the coordinate range calculator utility (`mmcv/utils/coordinate_range_calculator.py`):
```python
# Original range in LiDAR coordinates
original_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

# With accurate nuScenes sensor mounting (mean values)
# lidar2ego_translation = [0.9652, 0.0000, 1.8403]
# After transformation to ego + axis rotation:
point_cloud_range = [-30.0, -16.0, -0.16, 30.0, 16.0, 3.84]

# Detection post-processing ranges (with 5m margin for x,y)
post_center_range = [-35.0, -21.0, -0.66, 35.0, 21.0, 4.34]

# Map post-processing range (2D, 8 values)
post_center_range_map = [-35.0, -21.0, -35.0, -21.0, 35.0, 21.0, 35.0, 21.0]
```

The symmetric Y range [-16.0, 16.0] is chosen as:
- The closest integer values to the calculated range [-15.97, 14.03]
- Provides a clean, symmetric range for easier computation
- Maintains full coverage of the original perception area

## Bug Fixes Applied

### 1. Goal Direction Class Transformation
**Issue**: The code attempted to transform goal direction classes (scalar values 0-9) as if they were coordinates.

**Fix**:
```python
# Direction classes represent 45-degree sectors
# Rotate by 90 degrees = add 2 classes
gt_fut_goal_new = gt_fut_goal.copy()
non_static_mask = gt_fut_goal < 9  # Class 9 = static
gt_fut_goal_new[non_static_mask] = (gt_fut_goal[non_static_mask] + 2) % 8
```

### 2. CAN Bus Data Transformation
**Issue**: Velocity and acceleration vectors were not transformed to the new coordinate system.

**Fix**:
```python
# Transform velocity (can_bus[10:13])
vel_x, vel_y, vel_z = can_bus[10:13]
can_bus[10] = vel_y    # Y -> X
can_bus[11] = -vel_x   # -X -> Y

# Transform acceleration (can_bus[13:16])
acc_x, acc_y, acc_z = can_bus[13:16]
can_bus[13] = acc_y    # Y -> X
can_bus[14] = -acc_x   # -X -> Y
```

### 3. Dynamic Rotation Center Calculation
**Issue**: The BEV rotation center was hardcoded to [100, 100], assuming a specific BEV grid size.

**Fix**: Calculate rotation center dynamically based on actual BEV dimensions:
```python
# In VAD_transformer.py and transformer.py
rotate_center = [bev_h // 2, bev_w // 2]
```
This ensures correct rotation regardless of BEV grid size (100x100 or 200x200).

## Training Pipeline

### 5. BEV Feature Transformation

#### 5.1 Shift Calculation for Temporal Alignment

```python
# File: mmcv/models/modules/VAD_transformer.py, Line: 239-262
# Obtain rotation angle and shift with ego motion
delta_x = np.array([each['can_bus'][0] for each in kwargs['img_metas']])
delta_y = np.array([each['can_bus'][1] for each in kwargs['img_metas']])
ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])

# Calculate shift components
grid_length_y = grid_length[0]  # H direction (Y)
grid_length_x = grid_length[1]  # W direction (X)
translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
bev_angle = ego_angle - translation_angle

shift_x = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_x / bev_w
shift_y = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_y / bev_h
```

#### 5.2 BEV Rotation for Temporal Fusion

```python
# File: mmcv/models/modules/VAD_transformer.py, Line: 269-279
if self.rotate_prev_bev:
    for i in range(bs):
        rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
        tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
        tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                              center=[bev_h // 2, bev_w // 2])
        tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
        prev_bev[:, i] = tmp_prev_bev[:, 0]
```

## Inference and Output

### 6. Model Output Processing

#### 6.1 Detection Output Format

```python
# File: mmcv/models/dense_heads/VAD_head.py
# Model outputs are in new ego coordinate system:
# - Box centers: (x, y, z) in new ego coordinates
# - Box dimensions: (w, l, h) with swapped w/l
# - Box yaw: angle in new coordinate system
# - Velocities: (vx, vy) in new ego coordinates
```

### 7. Output to nuScenes Format (output_to_nusc_box)

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 2124-2157
def output_to_nusc_box(detection):
    '''Convert the output to the box class in the nuScenes.'''
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # Transform yaw for nuScenes convention
    # Original: box_yaw = -box_yaw - np.pi / 2
    # With axis rotation applied: we subtracted π/2 in transform_yaw_to_ego_new
    # So to get back to nuScenes convention: -(yaw_new + π/2) = -box_yaw - np.pi / 2
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        box = CustomNuscenesBox(
            center=box_gravity_center[i],
            size=box_dims[i],
            orientation=quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list
```

## Evaluation Pipeline

### 8. Coordinate Transformation for Evaluation (lidar_nusc_box_to_global)

```python
# File: mmcv/datasets/nuscenes_vad_dataset.py, Line: 2181-2211
def lidar_nusc_box_to_global(info, boxes, classes, eval_configs, eval_version='detection_cvpr_2019'):
    '''Convert the box from ego to global coordinate.'''
    box_list = []

    # Get inverse axis rotation matrix
    # Original: Y->X, -X->Y, so inverse is: X->-Y, Y->X
    inverse_axis_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)

    for box in boxes:
        # Transform center from new ego to standard ego coordinates
        center_in_new_coords = box.center
        box.center = center_in_new_coords @ inverse_axis_rotation.T

        # Transform velocity from new ego to standard ego coordinates
        if hasattr(box, 'velocity') and box.velocity is not None:
            vel_array = np.array(box.velocity)
            vel_transformed = vel_array @ inverse_axis_rotation.T
            box.velocity = tuple(vel_transformed)

        # Filter detection in ego coordinates
        x_distance, y_distance = box.center[0], box.center[1]
        if abs(x_distance) > det_range_x or abs(y_distance) > det_range_y:
            continue

        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)

    return box_list
```

## TensorRT Deployment

When deploying with TensorRT, ensure:

1. **Input Preprocessing**: Transform all inputs to the new coordinate system
2. **Camera Matrices**: Use ego_new2img instead of lidar2img
3. **CAN Bus Data**: Apply coordinate transformation to velocity/acceleration
4. **Output Interpretation**: Model outputs are in the new coordinate system

### 9.1 Export Configuration Update

```python
# File: trt/export_eval/patch/apply_coordinate_transform.py, Line: 16-44
def update_model_config(model):
    '''Update model configuration for new coordinate system.'''
    if hasattr(model, 'pts_bbox_head'):
        head = model.pts_bbox_head

        # Update pc_range
        if hasattr(head, 'pc_range'):
            head.pc_range = NEW_COORD_CONFIG['point_cloud_range']

        # Update bbox_coder
        if hasattr(head, 'bbox_coder') and hasattr(head.bbox_coder, 'pc_range'):
            head.bbox_coder.pc_range = NEW_COORD_CONFIG['point_cloud_range']
```

### 9.2 Preprocessing for TRT

```python
# File: trt/export_eval/patch/coordinate_transform.py, Line: 20-63
def transform_can_bus_to_ego_new(can_bus):
    '''Transform CAN bus data to new ego coordinate system.'''
    can_bus = can_bus.copy()

    # Transform velocity (indices 10-12)
    vel = can_bus[10:13].copy()
    can_bus[10] = vel[1]    # Y -> X
    can_bus[11] = -vel[0]   # -X -> Y

    # Transform acceleration (indices 13-15)
    acc = can_bus[13:16].copy()
    can_bus[13] = acc[1]    # Y -> X
    can_bus[14] = -acc[0]   # -X -> Y

    # Adjust yaw angle
    can_bus[16] -= np.pi / 2   # Subtract 90 degrees in radians
    can_bus[17] -= 90          # Subtract 90 degrees in degrees

    return can_bus
```

### 9.3 Shift Calculation for TRT

```python
# File: trt/export_eval/patch/coordinate_transform.py, Line: 65-107
def calculate_shift_ego_new(can_bus, grid_length=None, bev_h=100, bev_w=100):
    '''Calculate shift for BEV feature alignment in new coordinate system.'''
    # Calculate grid_length dynamically if not provided
    if grid_length is None:
        grid_length = calculate_grid_length(bev_w, bev_h)

    # grid_length is in [H, W] format, so [grid_length_y, grid_length_x]
    grid_length_y, grid_length_x = grid_length

    delta_x = can_bus[0]  # ego motion in x
    delta_y = can_bus[1]  # ego motion in y
    ego_angle = can_bus[-2]  # rotation angle (already adjusted)

    # In new coordinate system
    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    bev_angle = ego_angle / np.pi * 180 - translation_angle

    # Calculate shift components
    shift_x = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_x / bev_w
    shift_y = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_y / bev_h

    return np.array([shift_x, shift_y], dtype=np.float32)
```

### Validation Checklist

- [x] Model trains without errors
- [x] Loss values are reasonable
- [x] Bounding boxes align with camera images
- [ ] Map elements render correctly
- [ ] Trajectory predictions are smooth
- [ ] Evaluation metrics match expected values

## BEV Grid Considerations

With the coordinate transformation, the BEV grid now covers:
- **X-axis**: 60m range (was 30m)
- **Y-axis**: 32m range (was 60m)

For square BEV grids:
- **100x100 grid**: X resolution = 0.6m/pixel, Y resolution = 0.32m/pixel
- **200x200 grid**: X resolution = 0.3m/pixel, Y resolution = 0.16m/pixel

The different resolutions in X and Y are acceptable and maintain compatibility with existing model architectures.

## Benefits

1. **Standardization**: Aligns with common autonomous driving conventions
2. **Integration**: Easier integration with ego-centric planning systems
3. **Flexibility**: Transformation approach allows easy adaptation to other coordinate systems
4. **Performance**: No runtime overhead after initial data loading
5. **Accuracy**: Precise coordinate ranges based on actual sensor mounting parameters

## Future Improvements

1. Add configuration option to choose coordinate system
2. Implement automatic coordinate system detection
3. Create visualization tools for debugging transformations
4. Add unit tests for all transformation functions

## Complete Transformation Checklist

### Data Loading Stage
- [x] Point cloud coordinates: LiDAR → Ego → New Ego
- [x] Bounding box centers: LiDAR → Ego → New Ego
- [x] Bounding box dimensions: Swap width and length
- [x] Bounding box yaw angles: Subtract π/2
- [x] Velocities: Transform through both rotations
- [x] Future trajectories: Transform all waypoints
- [x] Goal direction classes: Add 2 (90°/45°)
- [x] Map elements: Transform all vertices
- [x] CAN bus velocity: Y→X, -X→Y
- [x] CAN bus acceleration: Y→X, -X→Y
- [x] CAN bus yaw angles: Subtract π/2 (both radians and degrees)
- [x] Camera matrices: ego_new → lidar → camera

### Training Stage
- [x] BEV shift calculation: Uses transformed ego motion
- [x] BEV rotation: Uses transformed rotation angle
- [x] All model operations: In new coordinate system

### Output Stage
- [x] Detection boxes: In new ego coordinates
- [x] Velocities: In new ego coordinates
- [x] Yaw angles: Adjusted for nuScenes convention

### Evaluation Stage
- [x] Box centers: Transform back to standard ego
- [x] Box velocities: Transform back to standard ego
- [x] Box orientations: Already in correct format
- [x] Global transformation: Standard ego → global

### TensorRT Deployment
- [x] Model configuration: Updated pc_range
- [x] Input preprocessing: Transform CAN bus data
- [x] Shift calculation: Adapted for new coordinates
- [x] Camera matrices: Transform lidar2img to ego_new2img

## Key Insights

1. **Consistency**: All data is transformed to the new coordinate system during loading and remains in this system throughout training and inference.

2. **Efficiency**: Transformations are applied once during data loading rather than repeatedly during training.

3. **Evaluation Compatibility**: The output is transformed back to the original coordinate system only for evaluation metrics.

4. **Dimension Swapping**: Due to the 90° rotation, bounding box width and length are swapped to maintain correct object dimensions.

5. **Angle Adjustments**: All angles (yaw, goal directions) are adjusted by ±90° depending on the context.

6. **Vector Transformations**: All vector quantities (positions, velocities, accelerations) undergo the same rotation transformation.

This comprehensive transformation ensures that the model can work in a more intuitive ego-centered coordinate system while maintaining compatibility with the standard nuScenes evaluation pipeline.
