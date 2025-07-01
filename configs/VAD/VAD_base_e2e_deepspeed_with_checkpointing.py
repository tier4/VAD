_base_ = ['./VAD_base_e2e_deepspeed.py']

# Enable activation checkpointing for BEVFormer layers
# This will reduce memory usage at the cost of slightly increased computation
model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            encoder=dict(
                transformerlayers=dict(
                    use_checkpoint=True  # Enable activation checkpointing
                )
            )
        )
    )
)

# With activation checkpointing, we can potentially increase batch size further
data = dict(
    samples_per_gpu=12,  # Further increased due to activation checkpointing
    workers_per_gpu=6    # Reduced workers to balance system resources
)

# Adjust learning rate for even larger batch
# Original lr = 4e-4 for batch_size 4
# New effective batch = 12 * 2 = 24 (6x larger)
optimizer = dict(
    type='AdamW',
    lr=4e-4 * 6,  # Scale by 6x
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)