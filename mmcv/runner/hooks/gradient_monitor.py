import torch
from .hook import HOOKS, Hook


@HOOKS.register_module()
class GradientMonitorHook(Hook):
    """Monitor gradient norms during training to diagnose overflow issues.
    
    Args:
        interval (int): Logging interval (every n iterations).
        monitor_grads (bool): Whether to monitor individual layer gradients.
        detect_anomaly (bool): Whether to enable PyTorch anomaly detection.
    """

    def __init__(self, 
                 interval=50, 
                 monitor_grads=True,
                 detect_anomaly=False):
        self.interval = interval
        self.monitor_grads = monitor_grads
        self.detect_anomaly = detect_anomaly
        
    def before_run(self, runner):
        """Enable anomaly detection if requested."""
        if self.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            runner.logger.info("PyTorch anomaly detection enabled")
    
    def after_train_iter(self, runner):
        """Log gradient statistics after each training iteration."""
        if self.every_n_iters(runner, self.interval):
            if self.monitor_grads:
                grad_norms = {}
                max_grad_norm = 0
                max_grad_layer = ""
                
                for name, param in runner.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_norms[name] = grad_norm
                        
                        # Check for NaN or Inf
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            runner.logger.warning(f"NaN/Inf detected in gradients of {name}")
                        
                        # Track maximum gradient
                        if grad_norm > max_grad_norm:
                            max_grad_norm = grad_norm
                            max_grad_layer = name
                
                # Log summary statistics
                runner.logger.info(
                    f"[Gradient Monitor] Iter {runner.iter}: "
                    f"Max grad norm: {max_grad_norm:.4f} in layer: {max_grad_layer}"
                )
                
                # Log top 5 layers with highest gradients
                if grad_norms:
                    sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
                    runner.logger.info("Top 5 gradient norms:")
                    for name, norm in sorted_grads:
                        runner.logger.info(f"  {name}: {norm:.4f}")
                
                # Check for potential overflow conditions
                if max_grad_norm > 1000:
                    runner.logger.warning(
                        f"Very large gradient detected: {max_grad_norm:.4f} in {max_grad_layer}. "
                        "This may cause FP16 overflow."
                    )