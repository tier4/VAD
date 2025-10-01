#!/usr/bin/env python
"""
Test script to verify config_utils functions with VAD_tiny_carla_tier4.py configuration.
"""

import sys
import os
sys.path.append('/mnt/sda1/VAD')

from mmcv import Config
from config_utils import (
    get_bev_dimensions,
    get_voxel_sizes,
    get_grid_length,
    get_class_counts,
    get_coordinate_system,
    get_model_variant_name,
    get_engine_paths,
    print_config_summary
)

def test_standard_config():
    """Test with standard VAD-tiny config."""
    print("\n" + "="*60)
    print("Testing Standard VAD-tiny Configuration")
    print("="*60)
    
    config_path = "/mnt/sda1/VAD/configs/VAD/VAD_tiny_e2e.py"
    if os.path.exists(config_path):
        cfg = Config.fromfile(config_path)
        print_config_summary(cfg)
        
        # Verify values
        bev_h, bev_w = get_bev_dimensions(cfg)
        assert bev_h == 117 and bev_w == 58, f"BEV dimensions mismatch: {bev_h}x{bev_w}"
        
        num_classes, map_num_classes = get_class_counts(cfg)
        assert num_classes == 10, f"Object classes mismatch: {num_classes}"
        assert map_num_classes == 3, f"Map classes mismatch: {map_num_classes}"
        
        coord_system = get_coordinate_system(cfg)
        assert coord_system == 'standard', f"Coordinate system mismatch: {coord_system}"
        
        print("✓ Standard config test passed")
    else:
        print(f"Config file not found: {config_path}")

def test_tier4_config():
    """Test with Tier4 CARLA config."""
    print("\n" + "="*60)
    print("Testing Tier4 CARLA Configuration")
    print("="*60)
    
    config_path = "/home/binwang/sda1/VAD/configs/VAD/VAD_tiny_carla_tier4.py"
    if os.path.exists(config_path):
        cfg = Config.fromfile(config_path)
        print_config_summary(cfg)
        
        # Verify values
        bev_h, bev_w = get_bev_dimensions(cfg)
        print(f"Expected BEV: 64x120, Got: {bev_h}x{bev_w}")
        
        num_classes, map_num_classes = get_class_counts(cfg)
        assert num_classes == 9, f"Object classes mismatch: {num_classes}"
        assert map_num_classes == 6, f"Map classes mismatch: {map_num_classes}"
        
        coord_system = get_coordinate_system(cfg)
        assert coord_system == 'tier4', f"Coordinate system mismatch: {coord_system}"
        
        model_variant = get_model_variant_name(cfg)
        assert 'tier4' in model_variant, f"Model variant should contain 'tier4': {model_variant}"
        
        engine_paths = get_engine_paths(cfg)
        print(f"Engine paths for {model_variant}:")
        for key, path in engine_paths.items():
            print(f"  {key}: {path}")
        
        print("✓ Tier4 config test passed")
    else:
        print(f"Config file not found: {config_path}")

def test_alternate_tier4_config():
    """Test with alternate Tier4 CARLA config path."""
    print("\n" + "="*60)
    print("Testing Alternate Tier4 CARLA Configuration Path")
    print("="*60)
    
    config_path = "/mnt/sda1/VAD/configs/VAD/VAD_tiny_carla_tier4.py"
    if os.path.exists(config_path):
        cfg = Config.fromfile(config_path)
        print_config_summary(cfg)
        
        # Verify values
        bev_h, bev_w = get_bev_dimensions(cfg)
        print(f"Expected BEV: 64x120, Got: {bev_h}x{bev_w}")
        
        num_classes, map_num_classes = get_class_counts(cfg)
        assert num_classes == 9, f"Object classes mismatch: {num_classes}"
        assert map_num_classes == 6, f"Map classes mismatch: {map_num_classes}"
        
        coord_system = get_coordinate_system(cfg)
        assert coord_system == 'tier4', f"Coordinate system mismatch: {coord_system}"
        
        model_variant = get_model_variant_name(cfg)
        assert 'tier4' in model_variant, f"Model variant should contain 'tier4': {model_variant}"
        
        print("✓ Alternate Tier4 config test passed")
    else:
        print(f"Config file not found: {config_path}")

if __name__ == "__main__":
    print("Testing config_utils with different VAD configurations...")
    
    # Test standard config
    test_standard_config()
    
    # Test Tier4 configs
    test_tier4_config()
    test_alternate_tier4_config()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)