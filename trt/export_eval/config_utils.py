# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for extracting TRT-relevant parameters from VAD config files.
Supports multiple VAD configurations including nuScenes, CARLA/B2D, and Tier4 variants.
"""

import os
from pathlib import Path


def get_bev_dimensions(cfg):
    """
    Extract BEV dimensions from config.
    
    Args:
        cfg: Config object from mmcv.Config.fromfile()
    
    Returns:
        tuple: (bev_h, bev_w) BEV height and width
    """
    # Check if directly specified in pts_bbox_head
    if hasattr(cfg.model.pts_bbox_head, 'bev_h') and hasattr(cfg.model.pts_bbox_head, 'bev_w'):
        return cfg.model.pts_bbox_head.bev_h, cfg.model.pts_bbox_head.bev_w
    
    # Calculate from grid_size if available
    if hasattr(cfg, 'grid_size'):
        # BEV dimensions are grid_size[1] for height and grid_size[0] for width
        return cfg.grid_size[1], cfg.grid_size[0]
    
    # Try to get from global variables if defined
    if hasattr(cfg, 'bev_h_') and hasattr(cfg, 'bev_w_'):
        return cfg.bev_h_, cfg.bev_w_
    
    # Default fallback for standard VAD-tiny
    print("Warning: Could not extract BEV dimensions from config, using default 100x100")
    return 100, 100


def get_voxel_sizes(cfg):
    """
    Extract voxel sizes from config.
    
    Args:
        cfg: Config object from mmcv.Config.fromfile()
    
    Returns:
        list: [voxel_x, voxel_y, voxel_z] voxel sizes
    """
    if hasattr(cfg, 'voxel_size'):
        return cfg.voxel_size
    
    # Default fallback
    print("Warning: Could not extract voxel_size from config, using default [0.6, 0.6, 4]")
    return [0.6, 0.6, 4]


def get_grid_length(cfg):
    """
    Calculate grid length from voxel sizes and point cloud range.
    Grid length is the physical size of each BEV grid cell.
    
    Args:
        cfg: Config object from mmcv.Config.fromfile()
    
    Returns:
        list: [grid_length_x, grid_length_y] in meters
    """
    if hasattr(cfg, 'point_cloud_range') and hasattr(cfg, 'voxel_size'):
        pc_range = cfg.point_cloud_range
        voxel_size = cfg.voxel_size
        
        # For VAD, grid_length typically equals voxel_size for x and y
        # But we should verify with BEV dimensions
        bev_h, bev_w = get_bev_dimensions(cfg)
        
        # Calculate actual grid length based on point cloud range and BEV dimensions
        grid_length_x = (pc_range[3] - pc_range[0]) / bev_w
        grid_length_y = (pc_range[4] - pc_range[1]) / bev_h
        
        return [grid_length_x, grid_length_y]
    
    # Default fallback for standard VAD-tiny
    print("Warning: Could not calculate grid_length from config, using default [0.6, 0.3]")
    return [0.6, 0.3]


def get_class_counts(cfg):
    """
    Extract object and map class counts from config.
    
    Args:
        cfg: Config object from mmcv.Config.fromfile()
    
    Returns:
        tuple: (num_classes, map_num_classes) number of object and map classes
    """
    # Object classes
    if hasattr(cfg, 'num_classes'):
        num_classes = cfg.num_classes
    elif hasattr(cfg, 'class_names'):
        num_classes = len(cfg.class_names)
    else:
        print("Warning: Could not extract num_classes from config, using default 10")
        num_classes = 10
    
    # Map classes
    if hasattr(cfg, 'map_num_classes'):
        map_num_classes = cfg.map_num_classes
    elif hasattr(cfg, 'map_classes'):
        map_num_classes = len(cfg.map_classes)
    else:
        print("Warning: Could not extract map_num_classes from config, using default 3")
        map_num_classes = 3
    
    return num_classes, map_num_classes


def get_coordinate_system(cfg):
    """
    Detect which coordinate system is being used.
    
    Args:
        cfg: Config object from mmcv.Config.fromfile()
    
    Returns:
        str: 'tier4' for Tier4 coordinate system, 'standard' for default
    """
    # Check dataset type for Tier4 indicator
    if hasattr(cfg.data, 'train') and hasattr(cfg.data.train, 'type'):
        if 'Tier4' in cfg.data.train.type or 'tier4' in cfg.data.train.type.lower():
            return 'tier4'
    
    # Check if config file name contains tier4
    if hasattr(cfg, 'filename'):
        if 'tier4' in cfg.filename.lower():
            return 'tier4'
    
    # Check point cloud range - Tier4 uses specific range
    if hasattr(cfg, 'point_cloud_range'):
        pc_range = cfg.point_cloud_range
        # Tier4 specific range: [-30.0, -16.0, -0.16, 30.0, 16.0, 3.84]
        if (pc_range[0] == -30.0 and pc_range[1] == -16.0 and 
            pc_range[3] == 30.0 and pc_range[4] == 16.0):
            return 'tier4'
    
    return 'standard'


def get_model_variant_name(cfg):
    """
    Generate a model variant name based on config characteristics.
    Used for organizing engine files.
    
    Args:
        cfg: Config object from mmcv.Config.fromfile()
    
    Returns:
        str: Model variant name (e.g., 'nuscenes_tiny', 'carla_tier4_tiny')
    """
    variant_parts = []
    
    # Dataset type
    if hasattr(cfg.data, 'train') and hasattr(cfg.data.train, 'type'):
        dataset_type = cfg.data.train.type
        if 'B2D' in dataset_type or 'CARLA' in dataset_type:
            variant_parts.append('carla')
        elif 'NuScenes' in dataset_type:
            variant_parts.append('nuscenes')
        else:
            variant_parts.append('custom')
    else:
        variant_parts.append('unknown')
    
    # Coordinate system
    coord_system = get_coordinate_system(cfg)
    if coord_system == 'tier4':
        variant_parts.append('tier4')
    
    # Model size (based on BEV dimensions)
    bev_h, bev_w = get_bev_dimensions(cfg)
    if bev_h <= 100 and bev_w <= 100:
        variant_parts.append('tiny')
    elif bev_h <= 200 and bev_w <= 200:
        variant_parts.append('base')
    else:
        variant_parts.append('large')
    
    return '_'.join(variant_parts)


def get_engine_paths(cfg, base_dir='scratch'):
    """
    Get engine file paths based on config.
    
    Args:
        cfg: Config object from mmcv.Config.fromfile()
        base_dir: Base directory for engine files
    
    Returns:
        dict: Dictionary with paths for different engines
    """
    variant = get_model_variant_name(cfg)
    base_path = Path(base_dir) / variant
    
    return {
        'extract_img_feat': base_path / 'vadv1.extract_img_feat' / 'sim_vadv1.extract_img_feat_fp16.engine',
        'pts_bbox_head_forward': base_path / 'vadv1.pts_bbox_head.forward' / 'sim_vadv1.pts_bbox_head.forward.engine',
        'pts_bbox_head_forward_prev': base_path / 'vadv1_prev.pts_bbox_head.forward' / 'sim_vadv1_prev.pts_bbox_head.forward.engine',
    }


def print_config_summary(cfg):
    """
    Print a summary of TRT-relevant config parameters.
    
    Args:
        cfg: Config object from mmcv.Config.fromfile()
    """
    bev_h, bev_w = get_bev_dimensions(cfg)
    voxel_sizes = get_voxel_sizes(cfg)
    grid_length = get_grid_length(cfg)
    num_classes, map_num_classes = get_class_counts(cfg)
    coord_system = get_coordinate_system(cfg)
    variant = get_model_variant_name(cfg)
    
    print("="*60)
    print("TRT Configuration Summary")
    print("="*60)
    print(f"Model Variant: {variant}")
    print(f"Coordinate System: {coord_system}")
    print(f"BEV Dimensions: {bev_h} x {bev_w}")
    print(f"Voxel Sizes: {voxel_sizes}")
    print(f"Grid Length: {grid_length}")
    print(f"Object Classes: {num_classes}")
    print(f"Map Classes: {map_num_classes}")
    
    if hasattr(cfg, 'point_cloud_range'):
        print(f"Point Cloud Range: {cfg.point_cloud_range}")
    
    print("="*60)