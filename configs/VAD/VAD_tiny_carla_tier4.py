"""
VAD tiny end-to-end training for CARLA/B2D with Tier4 coordinate system.

This configuration uses the Tier4 coordinate system with transformation (Y->X, -X->Y).
The coordinate origin is at the center of the rear axis, which is lower than typical LiDAR mount.
"""

_base_ = [
    '../_base_/default_runtime.py'
]

# =============================================================================
# Coordinate System Configuration - Tier4 Format
# =============================================================================
# Tier4 coordinate system with rear axis center as origin
# Y->X, -X->Y transformation applied

# LiDAR to ego translation (from LiDAR sensor to ground projection of rear axis)
# These values define the sensor mounting position relative to the ground below rear axis
wheel_radius = 0.305  # Wheel radius in meters
lidar_height_above_axis = 1.84  # LiDAR height above rear axis center
lidar2ego_translation = [0.39, 0.0, lidar_height_above_axis + wheel_radius]  # [x, y, z] in meters
# X: 0.39m - LiDAR is 0.39m forward from the rear axis center
# Y: 0.0m - LiDAR is centered laterally (no left/right offset)
# Z: 2.145m - LiDAR is above ground (1.84m above axis + 0.305m wheel radius)

# Dynamically compute point cloud range based on lidar2ego_translation
# LiDAR can see ±2.5m from its height, translated to ground origin
point_cloud_range = [-30.0, -16.0,
                     lidar2ego_translation[2] - 2.5,  # Min Z: LiDAR height - 2.5m
                     30.0, 16.0,
                     lidar2ego_translation[2] + 2.5]   # Max Z: LiDAR height + 2.5m

# Post-processing center range for filtering predictions (5m padding on XY, 0.5m on Z)
detection_post_center_range = [point_cloud_range[0] - 5.0,  # X min - 5m padding
                                point_cloud_range[1] - 5.0,  # Y min - 5m padding
                                point_cloud_range[2] - 0.5,  # Z min - 0.5m padding
                                point_cloud_range[3] + 5.0,  # X max + 5m padding
                                point_cloud_range[4] + 5.0,  # Y max + 5m padding
                                point_cloud_range[5] + 0.5]  # Z max + 0.5m padding

# For map elements (8 values: x_min, y_min repeated for compatibility)
map_post_center_range = [detection_post_center_range[0], detection_post_center_range[1],  # x_min, y_min
                         detection_post_center_range[0], detection_post_center_range[1],  # repeated
                         detection_post_center_range[3], detection_post_center_range[4],  # x_max, y_max
                         detection_post_center_range[3], detection_post_center_range[4]]  # repeated

# Voxel size for BEV representation
voxel_size = [0.3, 0.3, 5]

# Calculate grid_size from point_cloud_range and voxel_size
grid_size = [
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]), 
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])  
]

# Calculate BEV grid dimensions dynamically from grid_size
# Note: BEV dimensions are grid_size[1] for height and grid_size[0] for width
bev_h_ = grid_size[1]  # Height dimension (Y-axis in BEV)
bev_w_ = grid_size[0]  # Width dimension (X-axis in BEV)

# =============================================================================
# Model Configuration
# =============================================================================
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# CARLA/B2D class names - adapted from B2D_VAD_Dataset
class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian', 'others'
]
num_classes = len(class_names)

# Map classes for CARLA/B2D - from B2D_VAD_Dataset
map_classes = ['Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

# Planning and temporal parameters
past_frames = 2  # For ego trajectory history
future_frames = 6  # For ego trajectory prediction
map_root = "data_b2d/bench2drive/maps"

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
queue_length = 3  # each sequence contains `queue_length` frames for BEV temporal context
total_epochs = 20

model = dict(
    type='VAD',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='VADHead',
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        tot_epoch=total_epochs,
        use_traj_lr_warmup=False,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_his_encoder=None,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        ego_fut_mode=6,
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.0),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.0,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.0),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.0,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.0),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.0,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.0),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.0,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        use_pe=True,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=300,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        map_num_vec=map_num_vec,
        map_num_classes=map_num_classes,
        map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
        map_num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='VADPerceptionTransformer',
            map_num_vec=map_num_vec,
            map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            map_decoder=dict(
                type='MapDetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.0),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=detection_post_center_range,
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=num_classes),
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=map_post_center_range,
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=map_num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_traj=dict(type='L1Loss', loss_weight=0.2),
        loss_traj_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.2),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0),
        loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=1.0, dis_thresh=1.0),
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.5)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=grid_size,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),
        map_assigner=dict(
            type='MapHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
            pc_range=point_cloud_range))))

# =============================================================================
# Dataset Configuration - Using Tier4 Dataset Class
# =============================================================================
dataset_type = 'B2D_VAD_DatasetTier4'  # Use Tier4 coordinate transformation
data_root = 'data_b2d/bench2drive/'
file_client_args = dict(backend='disk')

# Name mapping for B2D dataset classes - comprehensive mapping from reference
name_mapping = {
    # ================= vehicle =================
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.audi.etron": 'car',
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.dodge.charger_police": 'car',
    "vehicle.dodge.charger_police_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.nissan.patrol_2021": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.ford.crown": 'car',
    "vehicle.tesla.model3": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": 'car',
    # van
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    # ================= traffic sign ============
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',
    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    # =================== Construction ===========
    "static.prop.warningconstruction": 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",
    "static.prop.constructioncone": 'traffic_cone',
    # ================= pedestrian ==============
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0003": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0010": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
    "walker.pedestrian.0015": 'pedestrian',
    "walker.pedestrian.0016": 'pedestrian',
    "walker.pedestrian.0017": 'pedestrian',
    "walker.pedestrian.0018": 'pedestrian',
    "walker.pedestrian.0019": 'pedestrian',
    "walker.pedestrian.0020": 'pedestrian',
    "walker.pedestrian.0021": 'pedestrian',
    "walker.pedestrian.0022": 'pedestrian',
    "walker.pedestrian.0025": 'pedestrian',
    "walker.pedestrian.0027": 'pedestrian',
    "walker.pedestrian.0030": 'pedestrian',
    "walker.pedestrian.0031": 'pedestrian',
    "walker.pedestrian.0032": 'pedestrian',
    "walker.pedestrian.0034": 'pedestrian',
    "walker.pedestrian.0035": 'pedestrian',
    "walker.pedestrian.0041": 'pedestrian',
    "walker.pedestrian.0042": 'pedestrian',
    "walker.pedestrian.0046": 'pedestrian',
    "walker.pedestrian.0047": 'pedestrian',
    # ==========================================
    "static.prop.dirtdebris01": 'others',
    "static.prop.dirtdebris02": 'others',
}

# Evaluation configuration for B2D
eval_cfg = dict(
    dist_ths=[0.5, 1.0, 2.0, 4.0],
    dist_th_tp=2.0,
    min_recall=0.1,
    min_precision=0.1,
    mean_ap_weight=5,
    class_names=['car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian'],
    tp_metrics=['trans_err', 'scale_err', 'orient_err', 'vel_err'],
    err_name_maping={'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE', 'vel_err': 'mAVE', 'attr_err': 'mAAE'},
    class_range={'car': (50, 50), 'van': (50, 50), 'truck': (50, 50), 'bicycle': (40, 40),
                 'traffic_sign': (30, 30), 'traffic_cone': (30, 30), 'traffic_light': (30, 30), 'pedestrian': (40, 40)}
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='VADFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D',\
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='VADFormatBundle3D', class_names=class_names, with_label=False, with_ego=True),
            dict(type='CustomCollect3D',\
                 keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                       'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                       'ego_lcf_feat', 'gt_attr_labels'])])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data_b2d/infos/b2d_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        bev_size=(bev_h_, bev_w_),
        point_cloud_range=point_cloud_range,
        lidar2ego_translation=lidar2ego_translation,  # Pass translation config
        queue_length=queue_length,
        sample_interval=5,
        past_frames=past_frames,
        future_frames=future_frames,
        map_root=map_root,
        map_file='data_b2d/infos/b2d_map_infos.pkl',
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        name_mapping=name_mapping,
        eval_cfg=eval_cfg,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data_b2d/infos/b2d_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        point_cloud_range=point_cloud_range,
        lidar2ego_translation=lidar2ego_translation,  # Pass translation config
        queue_length=queue_length,
        sample_interval=5,
        past_frames=past_frames,
        future_frames=future_frames,
        map_root=map_root,
        map_file='data_b2d/infos/b2d_map_infos.pkl',
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        name_mapping=name_mapping,
        eval_cfg=eval_cfg,
        samples_per_gpu=1,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data_b2d/infos/b2d_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        point_cloud_range=point_cloud_range,
        lidar2ego_translation=lidar2ego_translation,  # Pass translation config
        queue_length=queue_length,
        sample_interval=5,
        past_frames=past_frames,
        future_frames=future_frames,
        map_root=map_root,
        map_file='data_b2d/infos/b2d_map_infos.pkl',
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        name_mapping=name_mapping,
        eval_cfg=eval_cfg,
        samples_per_gpu=1,
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# =============================================================================
# Training Configuration
# =============================================================================
optimizer = dict(
    type='AdamW',
    lr=6e-4,  # Reduced for stability with Tier4 coordinates
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=32,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

evaluation = dict(interval=total_epochs, pipeline=test_pipeline, metric='bbox', map_metric='chamfer')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

custom_hooks = [dict(type='CustomSetEpochInfoHook')]