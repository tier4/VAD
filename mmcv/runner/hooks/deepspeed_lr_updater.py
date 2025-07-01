"""DeepSpeed-compatible learning rate scheduler hooks."""

from .lr_updater import LrUpdaterHook
from .hook import HOOKS


@HOOKS.register_module()
class DeepSpeedLrUpdaterHook(LrUpdaterHook):
    """Learning rate scheduler hook for DeepSpeed training.
    
    This hook is a no-op when using DeepSpeed because DeepSpeed
    handles learning rate scheduling internally through its own
    configuration.
    """
    
    def before_run(self, runner):
        # Skip LR initialization when using DeepSpeed (optimizer is None)
        if runner.optimizer is None:
            runner.logger.info(
                "DeepSpeed is handling learning rate scheduling internally. "
                "Skipping MMCV LR scheduler initialization."
            )
            return
        
        # Fall back to parent implementation if not using DeepSpeed
        super().before_run(runner)
    
    def before_train_epoch(self, runner):
        # Skip LR updates when using DeepSpeed
        if runner.optimizer is None:
            return
        super().before_train_epoch(runner)
    
    def before_train_iter(self, runner):
        # Skip LR updates when using DeepSpeed
        if runner.optimizer is None:
            return
        super().before_train_iter(runner)
    
    def get_lr(self, runner, base_lr):
        # Return base_lr when using DeepSpeed
        if runner.optimizer is None:
            return base_lr
        return super().get_lr(runner, base_lr)


@HOOKS.register_module()
class DeepSpeedCosineAnnealingLrUpdaterHook(DeepSpeedLrUpdaterHook):
    """Cosine annealing learning rate scheduler for DeepSpeed.
    
    This is a placeholder that does nothing when DeepSpeed is active,
    allowing the configuration to remain compatible.
    """
    
    def __init__(self, **kwargs):
        # Store parameters but don't use them with DeepSpeed
        self.min_lr = kwargs.get('min_lr', 0)
        self.min_lr_ratio = kwargs.get('min_lr_ratio', 0)
        super().__init__()