_base_ = [
    './VAD_base_e2e.py'
]

# Adjust batch size for available GPU memory
data = dict(
    samples_per_gpu=4,  # Reduced for GPU memory constraints
    workers_per_gpu=0,  # Set to 0 to avoid multiprocessing pickle issues temporarily
)

# Use DeepSpeed optimizer hook
optimizer_config = dict(
    type='DeepSpeedOptimizerHook'
)

# Checkpoint configuration for DeepSpeed
checkpoint_config = dict(
    type='DeepSpeedCheckpointHook',
    interval=1,
    save_optimizer=True,
    out_dir='./work_dirs'
)

# Scale learning rate with batch size
# Original lr = 2e-4 for batch_size 4*8=32
# New lr = 2e-4 * 0.5 = 1e-4 for batch_size 2*8=16
# Simplified optimizer config for DeepSpeed (no paramwise_cfg)
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01
)

# Disable LR scheduling - DeepSpeed handles this internally
lr_config = None

# Optional: Add gradient accumulation if needed
# gradient_accumulation_steps = 2
