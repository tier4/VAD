# Comprehensive Fix for DeepSpeed Pickle Error

## Problem
"TypeError: cannot pickle 'dict_keys' object" when using multiprocess data loading with DeepSpeed.

## Root Cause
In Python 3, `dict.keys()` returns a dict_keys view object which is not pickleable. When PyTorch DataLoader creates worker processes, it needs to pickle the dataset object.

## Fixes Applied

### 1. Direct Fix - Convert dict_keys to list (line 935)
```python
# Before:
self.class_names = self.class_range_y.keys()

# After:
self.class_names = list(self.class_range_y.keys())
```

### 2. Ensure clean dict objects (lines 926-927)
```python
# Ensure these are regular dicts, not dict subclasses
self.class_range_x = dict(class_range_x) if isinstance(class_range_x, dict) else class_range_x
self.class_range_y = dict(class_range_y) if isinstance(class_range_y, dict) else class_range_y
```

### 3. Add __getstate__ to v1CustomDetectionConfig (lines 971-978)
```python
def __getstate__(self):
    """Ensure the object is pickleable by converting any dict_keys to lists"""
    state = self.__dict__.copy()
    for key, value in state.items():
        if type(value).__name__ == 'dict_keys':
            state[key] = list(value)
    return state
```

### 4. Add comprehensive __getstate__ to VADCustomNuScenesDataset (lines 1046-1073)
```python
def __getstate__(self):
    """Ensure the dataset is pickleable by converting any dict_keys to lists"""
    state = self.__dict__.copy()
    # Recursively check and fix dict_keys objects
    # ... (see code for full implementation)
    return state
```

### 5. Temporary Workaround (if needed)
Set `workers_per_gpu=0` in the config to disable multiprocessing:
```python
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,  # Disable multiprocessing
)
```

## Additional Steps Taken
1. Cleared all Python cache files (*.pyc and __pycache__)
2. Fixed DeepSpeed config to use actual values instead of "auto"
3. Created debugging scripts to identify the source of dict_keys

## Testing
The fixes should allow DeepSpeed training to work with multiprocess data loading. If issues persist, check:
1. External dependencies (nuscenes-devkit)
2. Custom data transforms in the pipeline
3. Other nested objects that might contain dict_keys