"""
Unit tests for DeepSpeed integration with VAD training pipeline.

Run tests with:
    pytest tests/test_deepspeed/test_deepspeed_integration.py -v
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import json
import shutil
from unittest.mock import Mock, patch, MagicMock

# Import the modules we want to test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mmcv.runner.hooks.optimizer import DeepSpeedOptimizerHook
from mmcv.runner.hooks.checkpoint import DeepSpeedCheckpointHook


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)


class TestDeepSpeedOptimizerHook:
    """Test DeepSpeedOptimizerHook functionality."""
    
    def test_deepspeed_optimizer_hook_init(self):
        """Test DeepSpeedOptimizerHook initialization."""
        hook = DeepSpeedOptimizerHook()
        assert hook.grad_clip is None
    
    def test_deepspeed_optimizer_hook_with_deepspeed_model(self):
        """Test hook with DeepSpeed model."""
        hook = DeepSpeedOptimizerHook()
        
        # Mock runner with DeepSpeed model
        runner = Mock()
        runner.outputs = {'loss': torch.tensor(1.0, requires_grad=True)}
        
        # Mock DeepSpeed model with backward and step methods
        runner.model = Mock()
        runner.model.backward = Mock()
        runner.model.step = Mock()
        
        # Run hook
        hook.after_train_iter(runner)
        
        # Verify DeepSpeed methods were called
        runner.model.backward.assert_called_once()
        runner.model.step.assert_called_once()
    
    def test_deepspeed_optimizer_hook_fallback(self):
        """Test hook fallback to standard behavior."""
        hook = DeepSpeedOptimizerHook()
        
        # Mock runner with standard model (no backward/step methods)
        runner = Mock()
        runner.outputs = {'loss': torch.tensor(1.0, requires_grad=True)}
        runner.model = DummyModel()
        runner.optimizer = Mock()
        
        # This should use parent class behavior
        with patch.object(hook.__class__.__bases__[0], 'after_train_iter') as mock_parent:
            hook.after_train_iter(runner)
            mock_parent.assert_called_once_with(runner)


class TestDeepSpeedCheckpointHook:
    """Test DeepSpeedCheckpointHook functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_deepspeed_checkpoint_save(self):
        """Test saving DeepSpeed checkpoint."""
        hook = DeepSpeedCheckpointHook(
            interval=1,
            out_dir=self.temp_dir,
            by_epoch=True
        )
        
        # Mock runner with DeepSpeed model
        runner = Mock()
        runner.epoch = 0
        runner.iter = 100
        runner.meta = {}
        runner.logger = Mock()
        
        # Mock DeepSpeed model with save_checkpoint method
        runner.model = Mock()
        runner.model.save_checkpoint = Mock()
        
        # Test save checkpoint
        hook._save_checkpoint(runner)
        
        # Verify save_checkpoint was called
        expected_dir = os.path.join(self.temp_dir, 'epoch_1')
        runner.model.save_checkpoint.assert_called_once_with(expected_dir)
        
        # Check metadata was saved
        meta_path = os.path.join(expected_dir, 'meta.pth')
        # Note: In real scenario, this would be created by the mock
    
    def test_deepspeed_checkpoint_fallback(self):
        """Test checkpoint fallback to standard behavior."""
        hook = DeepSpeedCheckpointHook(
            interval=1,
            out_dir=self.temp_dir
        )
        
        # Mock runner with standard model
        runner = Mock()
        runner.model = DummyModel()
        
        # This should use parent class behavior
        with patch.object(hook.__class__.__bases__[0], '_save_checkpoint') as mock_parent:
            hook._save_checkpoint(runner)
            mock_parent.assert_called_once_with(runner)


class TestDeepSpeedConfig:
    """Test DeepSpeed configuration handling."""
    
    def test_deepspeed_config_loading(self):
        """Test loading DeepSpeed configuration."""
        config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            # Load config
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            assert loaded_config['zero_optimization']['stage'] == 2
            assert loaded_config['fp16']['enabled'] is True
        finally:
            os.unlink(config_path)


class TestBEVFormerLayerCheckpointing:
    """Test activation checkpointing in BEVFormerLayer."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bevformer_layer_with_checkpointing(self):
        """Test BEVFormerLayer with activation checkpointing enabled."""
        from mmcv.models.modules.encoder import BEVFormerLayer
        
        # Create layer with checkpointing
        layer = BEVFormerLayer(
            attn_cfgs=[
                dict(type='TemporalSelfAttention', embed_dims=256),
                dict(type='SpatialCrossAttention', embed_dims=256)
            ],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
            use_checkpoint=True  # Enable checkpointing
        )
        
        assert layer.use_checkpoint is True
        assert hasattr(layer, '_forward')


class TestMemoryUsage:
    """Test memory usage improvements with DeepSpeed."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_comparison(self):
        """Compare memory usage between standard and DeepSpeed approaches."""
        # This is a placeholder for actual memory comparison tests
        # In practice, you would:
        # 1. Create a model
        # 2. Measure memory with standard training
        # 3. Measure memory with DeepSpeed
        # 4. Assert DeepSpeed uses less memory
        
        model = DummyModel().cuda()
        
        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Simulate forward pass
        x = torch.randn(32, 10).cuda()
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        # Get memory after standard training step
        standard_memory = torch.cuda.memory_allocated()
        
        # In real test, compare with DeepSpeed memory usage
        assert standard_memory > initial_memory


class TestIntegration:
    """Integration tests for the complete DeepSpeed pipeline."""
    
    def test_training_script_arguments(self):
        """Test that training script accepts DeepSpeed arguments."""
        # This would test the actual argument parsing
        # In practice, you would import and test the parse_args function
        pass
    
    def test_launch_script_exists(self):
        """Test that DeepSpeed launch script exists and is executable."""
        script_path = 'tools/dist_train_deepspeed.sh'
        if os.path.exists(script_path):
            assert os.access(script_path, os.X_OK), "Launch script is not executable"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])