import torch
import torch.nn as nn
from mmcv.models.builder import DETECTORS


def check_and_fix_tensor(tensor, name=""):
    """Check tensor for NaN/Inf and fix if needed."""
    if tensor is None:
        return tensor
        
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"WARNING: NaN/Inf detected in {name}")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Additional clamping for safety
    if tensor.dtype == torch.float16:
        tensor = torch.clamp(tensor, min=-65000, max=65000)
    
    return tensor


class FP16SafeWrapper(nn.Module):
    """Wrapper to add FP16 safety checks to model outputs."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        
        # Check and fix outputs
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    outputs[key] = check_and_fix_tensor(value, key)
                elif isinstance(value, (list, tuple)):
                    outputs[key] = [check_and_fix_tensor(v, f"{key}[{i}]") 
                                   for i, v in enumerate(value)]
        
        return outputs
    
    def __getattr__(self, name):
        # Forward attribute access to wrapped model
        if name == 'model':
            return super().__getattr__(name)
        return getattr(self.model, name)