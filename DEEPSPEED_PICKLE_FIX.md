# DeepSpeed Pickle Fix

## Bug 7: TypeError - cannot pickle 'dict_keys' object

### Error
```
TypeError: cannot pickle 'dict_keys' object
```

This error occurred when PyTorch DataLoader tried to create multiple worker processes for data loading.

### Root Cause
In `/home/ubuntu/work/VAD_Universe/mmcv/datasets/nuscenes_vad_dataset.py`, the `v1CustomDetectionConfig` class had this line:
```python
self.class_names = self.class_range_y.keys()
```

In Python 3, `.keys()` returns a `dict_keys` object which is a view object and not picklable. When PyTorch creates multiple worker processes for the DataLoader (when `workers_per_gpu > 0`), it needs to pickle the dataset object to send it to the worker processes.

### Fix
Changed line 935 in `nuscenes_vad_dataset.py`:
```python
# Before:
self.class_names = self.class_range_y.keys()

# After:
self.class_names = list(self.class_range_y.keys())
```

### Result
- The training now starts successfully without pickle errors
- Multi-process data loading works correctly
- The fix maintains the same functionality while ensuring compatibility with multiprocessing

### Additional Notes
This is a common issue when migrating from Python 2 to Python 3:
- Python 2: `dict.keys()` returns a list
- Python 3: `dict.keys()` returns a dict_keys view object

When using multiprocessing or any form of serialization (pickle), always convert dict_keys to list if you need to store them as attributes.