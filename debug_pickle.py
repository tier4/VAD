#!/usr/bin/env python3
"""
Debug pickle issues with the VAD dataset
"""
import pickle
import sys
import traceback

# Monkey-patch pickle to debug what's being pickled
original_dump = pickle.dump
original_dumps = pickle.dumps

def debug_dump(obj, file, protocol=None, *, fix_imports=True):
    """Wrap pickle.dump to catch dict_keys objects"""
    try:
        _check_object(obj, "")
        return original_dump(obj, file, protocol, fix_imports=fix_imports)
    except Exception as e:
        print(f"Pickle dump error: {e}")
        traceback.print_exc()
        raise

def debug_dumps(obj, protocol=None, *, fix_imports=True):
    """Wrap pickle.dumps to catch dict_keys objects"""
    try:
        _check_object(obj, "")
        return original_dumps(obj, protocol, fix_imports=fix_imports)
    except Exception as e:
        print(f"Pickle dumps error: {e}")
        traceback.print_exc()
        raise

def _check_object(obj, path):
    """Recursively check for dict_keys objects"""
    if type(obj).__name__ == 'dict_keys':
        raise TypeError(f"Found dict_keys object at path: {path}")
    
    if hasattr(obj, '__dict__'):
        for attr, value in obj.__dict__.items():
            _check_object(value, f"{path}.{attr}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _check_object(value, f"{path}[{repr(key)}]")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _check_object(value, f"{path}[{i}]")

# Apply the monkey-patch
pickle.dump = debug_dump
pickle.dumps = debug_dumps

# Now try to load and pickle the dataset
if __name__ == "__main__":
    print("Testing VAD dataset pickling...")
    
    import sys
    sys.path.insert(0, '/home/ubuntu/work/VAD_Universe')
    
    from mmcv.datasets import VADCustomNuScenesDataset
    from mmcv.utils.config import Config
    
    # Load config
    cfg = Config.fromfile('configs/VAD/VAD_base_e2e.py')
    
    # Create dataset with minimal config
    dataset_cfg = dict(
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/vad_nuscenes_infos_temporal_train.pkl',
        pipeline=[],
        classes=['car'],
        modality=dict(use_camera=True),
        test_mode=False,
        use_valid_flag=True,
        bev_size=(200, 200),
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        queue_length=4,
        map_classes=['divider'],
        map_fixed_ptsnum_per_line=20,
        map_eval_use_same_gt_sample_num_flag=True,
        box_type_3d='LiDAR',
        custom_eval_version='vad_nusc_detection_cvpr_2019'
    )
    
    try:
        print("Creating dataset...")
        dataset = VADCustomNuScenesDataset(**dataset_cfg)
        
        print("Testing pickle...")
        pickled = pickle.dumps(dataset)
        print("Success! Dataset can be pickled.")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()