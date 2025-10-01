"""
B2D VAD Dataset with Tier4 Coordinate System

This dataset applies the Tier4 coordinate transformation (Y->X, -X->Y) to B2D/CARLA data.
The coordinate origin is at the center of the rear axis, which is lower than typical LiDAR mount.
"""

import os
import copy
import numpy as np
import torch
from os import path as osp
from .builder import DATASETS
from .B2D_vad_dataset import B2D_VAD_Dataset
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity
from shapely.geometry import LineString


@DATASETS.register_module()
class B2D_VAD_DatasetTier4(B2D_VAD_Dataset):
    """
    B2D/CARLA Dataset with Tier4 coordinate system transformation.
    
    The Tier4 coordinate system applies a 90-degree counter-clockwise rotation:
    - Y axis becomes X axis  
    - -X axis becomes Y axis
    - Z axis remains unchanged
    
    The origin is at the center of the rear axis (lower than typical LiDAR mount).
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize B2D_VAD_DatasetTier4 with optional lidar2ego_translation.

        The lidar2ego_translation can be passed in kwargs:
            lidar2ego_translation: Optional [x, y, z] translation from LiDAR to ego frame.
                                  Default: [0.9652, 0.0000, 1.8403] for typical sensor mounting.
                                  If None, uses default values.
        """
        # Extract lidar2ego_translation from kwargs before passing to parent
        lidar2ego_translation = kwargs.pop('lidar2ego_translation', None)

        super().__init__(*args, **kwargs)

        # Use provided translation or default values
        if lidar2ego_translation is not None:
            self.lidar2ego_translation = np.array(lidar2ego_translation, dtype=np.float32)
        else:
            # Default: LiDAR position relative to ground projection of rear axis
            wheel_radius = 0.305  # Default wheel radius
            lidar_height_above_axis = 1.84  # LiDAR height above rear axis center
            # LiDAR is 0.39m forward, 0m lateral, 2.145m above ground (1.84m + 0.305m wheel radius)
            self.lidar2ego_translation = np.array([0.39, 0.0, lidar_height_above_axis + wheel_radius], dtype=np.float32)
        
    @staticmethod
    def get_axis_rotation_matrix():
        """Get the axis rotation matrix: Y->X, -X->Y, Z->Z"""
        return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    
    def transform_position_to_tier4(self, position, apply_translation=True):
        """
        Transform position from standard to Tier4 coordinates.
        This is a two-step process:
        1. Translate from LiDAR frame to ego frame (rear axis center)
        2. Apply axis rotation (Y->X, -X->Y)

        Args:
            position: (N, 2) or (N, 3) or (2,) or (3,) array of positions
            apply_translation: Whether to apply lidar2ego translation (default: True)
        Returns:
            Transformed positions in Tier4 coordinate system
        """
        axis_rotation = self.get_axis_rotation_matrix()

        if len(position.shape) == 1:
            # Single position vector
            if position.shape[0] == 2:
                # 2D position
                pos_3d = np.array([position[0], position[1], 0])

                # Step 1: Apply translation from LiDAR to ego (if enabled)
                if apply_translation:
                    pos_3d = pos_3d + self.lidar2ego_translation

                # Step 2: Apply axis rotation
                pos_new = pos_3d @ axis_rotation.T
                return pos_new[:2]
            else:
                # 3D position
                # Step 1: Apply translation from LiDAR to ego (if enabled)
                if apply_translation:
                    pos_ego = position + self.lidar2ego_translation
                else:
                    pos_ego = position

                # Step 2: Apply axis rotation
                return pos_ego @ axis_rotation.T
        else:
            # Multiple positions
            if position.shape[-1] == 2:
                # 2D positions
                pos_3d = np.concatenate([position, np.zeros((position.shape[0], 1))], axis=-1)

                # Step 1: Apply translation from LiDAR to ego (if enabled)
                if apply_translation:
                    pos_3d = pos_3d + self.lidar2ego_translation.reshape(1, 3)

                # Step 2: Apply axis rotation
                pos_new = pos_3d @ axis_rotation.T
                return pos_new[:, :2]
            else:
                # 3D positions
                # Step 1: Apply translation from LiDAR to ego (if enabled)
                if apply_translation:
                    pos_ego = position + self.lidar2ego_translation.reshape(1, 3)
                else:
                    pos_ego = position

                # Step 2: Apply axis rotation
                return pos_ego @ axis_rotation.T
    
    def transform_velocity_to_tier4(self, velocity):
        """
        Transform velocity from standard to Tier4 coordinates.
        Args:
            velocity: (2,) or (N, 2) velocity [vx, vy]
        Returns:
            Transformed velocity in Tier4 coordinate system
        """
        axis_rotation = self.get_axis_rotation_matrix()
        
        if len(velocity.shape) == 1:
            # Single velocity vector
            vel_3d = np.array([velocity[0], velocity[1], 0])
            vel_new = vel_3d @ axis_rotation.T
            return vel_new[:2]
        else:
            # Multiple velocities
            vel_3d = np.concatenate([velocity, np.zeros((velocity.shape[0], 1))], axis=-1)
            vel_new = vel_3d @ axis_rotation.T
            return vel_new[:, :2]
    
    def transform_world_to_tier4(self, position):
        """
        Transform world coordinates to Tier4 coordinate system.
        Only applies rotation, NO translation (since world origin doesn't change).

        Args:
            position: (N, 2) or (N, 3) or (2,) or (3,) array of world positions
        Returns:
            Rotated positions in Tier4 world coordinate system
        """
        axis_rotation = self.get_axis_rotation_matrix()

        if len(position.shape) == 1:
            # Single position vector
            if position.shape[0] == 2:
                # 2D position
                pos_3d = np.array([position[0], position[1], 0])
                pos_new = pos_3d @ axis_rotation.T
                return pos_new[:2]
            else:
                # 3D position
                return position @ axis_rotation.T
        else:
            # Multiple positions
            if position.shape[-1] == 2:
                # 2D positions
                pos_3d = np.concatenate([position, np.zeros((position.shape[0], 1))], axis=-1)
                pos_new = pos_3d @ axis_rotation.T
                return pos_new[:, :2]
            else:
                # 3D positions
                return position @ axis_rotation.T

    def transform_yaw_to_tier4(self, yaw):
        """
        Transform yaw angle from standard to Tier4 coordinates.
        Args:
            yaw: yaw angle(s) in standard coordinates
        Returns:
            Transformed yaw in Tier4 coordinate system
        """
        # Apply 90-degree rotation due to axis change (Y->X, -X->Y)
        # Coordinate system rotates counter-clockwise by 90°, so yaw decreases by 90°
        yaw_new = yaw - np.pi / 2
        
        # Normalize to [-pi, pi]
        yaw_new = np.arctan2(np.sin(yaw_new), np.cos(yaw_new))
        
        return yaw_new
    
    def transform_bbox_to_tier4(self, bbox):
        """
        Transform bounding box from standard to Tier4 coordinates.
        Args:
            bbox: Dictionary or object with center, size, velocity, yaw attributes
        Returns:
            Transformed bbox in Tier4 coordinate system
        """
        # Transform center position
        if hasattr(bbox, 'center'):
            center = np.array(bbox.center)
            bbox.center = self.transform_position_to_tier4(center).tolist()
        elif isinstance(bbox, dict) and 'translation' in bbox:
            center = np.array(bbox['translation'])
            bbox['translation'] = self.transform_position_to_tier4(center).tolist()
        
        # Transform velocity if present
        if hasattr(bbox, 'velocity'):
            vel = np.array(bbox.velocity[:2])  # Only 2D velocity
            new_vel = self.transform_velocity_to_tier4(vel)
            bbox.velocity = [new_vel[0], new_vel[1]] + bbox.velocity[2:]
        elif isinstance(bbox, dict) and 'velocity' in bbox:
            vel = np.array(bbox['velocity'][:2])
            new_vel = self.transform_velocity_to_tier4(vel)
            bbox['velocity'] = [new_vel[0], new_vel[1]] + bbox['velocity'][2:]
        
        # Transform yaw angle
        if hasattr(bbox, 'yaw'):
            bbox.yaw = self.transform_yaw_to_tier4(bbox.yaw)
        elif isinstance(bbox, dict) and 'rotation' in bbox:
            # Assuming rotation is in quaternion format, extract yaw
            yaw = quaternion_yaw(Quaternion(bbox['rotation']))
            new_yaw = self.transform_yaw_to_tier4(yaw)
            # Convert back to quaternion (simplified - only yaw rotation)
            bbox['rotation'] = [np.cos(new_yaw/2), 0, 0, np.sin(new_yaw/2)]
        
        # Note: Size/dimensions don't need transformation as they're object-relative
        
        return bbox
    
    def transform_trajectory_to_tier4(self, trajectory):
        """
        Transform trajectory points from standard to Tier4 coordinates.
        Args:
            trajectory: (N, 2) or (N, 3) array of trajectory points
        Returns:
            Transformed trajectory in Tier4 coordinate system
        """
        if trajectory is None or len(trajectory) == 0:
            return trajectory
        
        trajectory = np.array(trajectory)
        return self.transform_position_to_tier4(trajectory)
    
    def transform_polyline_to_tier4(self, polyline):
        """
        Transform map polyline from standard to Tier4 coordinates.
        Map polylines are already in world coordinates, so we only apply rotation.
        Args:
            polyline: List of (x, y) points or numpy array
        Returns:
            Transformed polyline in Tier4 coordinate system
        """
        if polyline is None or len(polyline) == 0:
            return polyline

        polyline = np.array(polyline)
        # Map polylines are in world coordinates, not ego-relative
        # Only apply rotation, not the LiDAR offset
        return self.transform_world_to_tier4(polyline)
    
    def get_data_info(self, index):
        """
        Get data info with Tier4 coordinate transformation applied.

        Overrides parent method to apply coordinate transformations to:
        - Bounding boxes (center, velocity, yaw)
        - Ego trajectories (past and future)
        - Map polylines
        - Camera order (to match Autoware order)
        """
        # Get base info from parent class - but we'll override camera processing
        import copy
        info = self.data_infos[index]

        # Process name mapping
        if hasattr(self, 'NameMapping') and self.NameMapping:
            for i in range(len(info['gt_names'])):
                if info['gt_names'][i] in self.NameMapping.keys():
                    info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]

        # Transform ego data to new coordinate system
        # ego_translation is in WORLD coordinates, only rotate axes, no translation
        ego_translation = np.array(info['ego_translation'])
        ego_translation_new = self.transform_world_to_tier4(ego_translation[:2])
        ego_translation_transformed = np.array([ego_translation_new[0], ego_translation_new[1], ego_translation[2]])

        # Velocity and acceleration are also world-relative, only rotate
        ego_vel = np.array(info['ego_vel'])
        ego_vel_new = self.transform_world_to_tier4(ego_vel[:2])
        ego_vel_transformed = np.array([ego_vel_new[0], ego_vel_new[1], ego_vel[2]])

        ego_accel = np.array(info['ego_accel'])
        ego_accel_new = self.transform_world_to_tier4(ego_accel[:2])
        ego_accel_transformed = np.array([ego_accel_new[0], ego_accel_new[1], ego_accel[2]])

        # Transform rotation rate (angular velocity)
        ego_rotation_rate = np.array(info['ego_rotation_rate'])
        # For 90° CCW rotation: omega_x_new = omega_y_old, omega_y_new = -omega_x_old, omega_z unchanged
        ego_rotation_rate_transformed = np.array([ego_rotation_rate[1], -ego_rotation_rate[0], ego_rotation_rate[2]])

        # Transform yaw
        ego_yaw = np.nan_to_num(info['ego_yaw'], nan=np.pi/2)
        ego_yaw_new = self.transform_yaw_to_tier4(ego_yaw)

        input_dict = dict(
            folder=info['folder'],
            scene_token=info['folder'],
            frame_idx=info['frame_idx'],
            ego_yaw=ego_yaw_new,
            ego_translation=ego_translation_transformed,
            sensors=info['sensors'],
            world2lidar=info['sensors']['LIDAR_TOP']['world2lidar'],
            gt_ids=info['gt_ids'],
            gt_boxes=info['gt_boxes'],
            gt_names=info['gt_names'],
            ego_vel=ego_vel_transformed,
            ego_accel=ego_accel_transformed,
            ego_rotation_rate=ego_rotation_rate_transformed,
            npc2world=info['npc2world'],
            timestamp=info['frame_idx']/10
        )

        # Process cameras in Autoware order
        if self.modality['use_camera']:
            # Define Autoware camera order
            camera_order = [
                'CAM_FRONT',       # Index 0
                'CAM_BACK',        # Index 1
                'CAM_FRONT_LEFT',  # Index 2
                'CAM_BACK_LEFT',   # Index 3
                'CAM_FRONT_RIGHT', # Index 4
                'CAM_BACK_RIGHT'   # Index 5
            ]

            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []

            lidar2ego_original = info['sensors']['LIDAR_TOP']['lidar2ego']
            lidar2global = self.invert_pose(info['sensors']['LIDAR_TOP']['world2lidar'])

            # Create transformation from new lidar coord (ground, rotated) to original lidar coord
            # This is the inverse of our Tier4 transformation
            new_lidar_to_old_lidar = np.eye(4)
            # Inverse rotation (transpose of rotation matrix)
            axis_rotation_inv = self.get_axis_rotation_matrix().T
            new_lidar_to_old_lidar[:3, :3] = axis_rotation_inv
            # Inverse translation (negative of our translation)
            # Note: No rotation needed as lidar2ego_translation is already in old LiDAR frame
            inv_translation = -self.lidar2ego_translation
            new_lidar_to_old_lidar[:3, 3] = inv_translation

            # Process cameras in the specified order
            for cam_name in camera_order:
                if cam_name in info['sensors']:
                    cam_info = info['sensors'][cam_name]
                    image_paths.append(osp.join(self.data_root, cam_info['data_path']))

                    # Obtain lidar to image transformation matrix
                    cam2ego = cam_info['cam2ego']
                    intrinsic = cam_info['intrinsic']
                    intrinsic_pad = np.eye(4)
                    intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                    # Transform from new lidar coord to cam:
                    # new_lidar -> old_lidar -> ego -> cam
                    lidar2cam = self.invert_pose(cam2ego) @ lidar2ego_original @ new_lidar_to_old_lidar
                    lidar2img = intrinsic_pad @ lidar2cam
                    lidar2img_rts.append(lidar2img)
                    cam_intrinsics.append(intrinsic_pad)
                    lidar2cam_rts.append(lidar2cam)

            # Update lidar2global for new coordinate system
            # new_lidar -> old_lidar -> global
            lidar2global_new = lidar2global @ new_lidar_to_old_lidar

            input_dict.update(dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                l2g_r_mat=lidar2global_new[0:3,0:3],
                l2g_t=lidar2global_new[0:3,3]
            ))

        # Get annotations
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        # Process ego information (using transformed values)
        yaw = input_dict['ego_yaw']  # Already transformed
        rotation = list(Quaternion(axis=[0, 0, 1], radians=yaw))

        if yaw < 0:
            yaw += 2*np.pi
        yaw_in_degree = yaw / np.pi * 180

        can_bus = np.zeros(18)
        can_bus[:3] = input_dict['ego_translation']  # Already transformed
        can_bus[3:7] = rotation
        can_bus[7:10] = input_dict['ego_vel']  # Already transformed
        can_bus[10:13] = input_dict['ego_accel']  # Already transformed
        can_bus[13:16] = input_dict['ego_rotation_rate']  # Already transformed
        can_bus[16] = yaw
        can_bus[17] = yaw_in_degree

        input_dict.update(dict(can_bus=can_bus))

        # Process ego_lcf_feat (same as parent)
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = input_dict['ego_translation'][0:2]
        ego_lcf_feat[2:4] = input_dict['ego_accel'][0:2]
        ego_lcf_feat[4] = input_dict['ego_rotation_rate'][-1]
        ego_lcf_feat[5] = info['ego_size'][1]
        ego_lcf_feat[6] = info['ego_size'][0]
        ego_lcf_feat[7] = np.sqrt(input_dict['ego_translation'][0]**2+input_dict['ego_translation'][1]**2)
        ego_lcf_feat[8] = info['steer']

        # Get ego trajectories using parent method
        ego_his_trajs, ego_fut_trajs, ego_fut_masks, command = self.get_ego_trajs(
            index, self.sample_interval, self.past_frames, self.future_frames
        )

        input_dict['ego_his_trajs'] = ego_his_trajs
        input_dict['ego_fut_trajs'] = ego_fut_trajs
        input_dict['ego_fut_masks'] = ego_fut_masks
        input_dict['ego_fut_cmd'] = command
        input_dict['ego_lcf_feat'] = ego_lcf_feat
        input_dict['fut_valid_flag'] = (ego_fut_masks==1).all()

        info = input_dict
        
        # Transform bounding boxes
        if 'gt_boxes' in info:
            for i in range(len(info['gt_boxes'])):
                info['gt_boxes'][i] = self.transform_bbox_to_tier4(info['gt_boxes'][i])

        # Transform ego trajectories (these are frame-to-frame OFFSETS, not positions)
        # Only rotate, don't translate
        if 'ego_his_trajs' in info:
            info['ego_his_trajs'] = self.transform_position_to_tier4(
                info['ego_his_trajs'], apply_translation=False
            )

        if 'ego_fut_trajs' in info:
            info['ego_fut_trajs'] = self.transform_position_to_tier4(
                info['ego_fut_trajs'], apply_translation=False
            )

        # Transform map polylines
        if 'gt_lines_instance' in info:
            transformed_lines = []
            for line in info['gt_lines_instance']:
                if isinstance(line, LineString):
                    # Transform LineString coordinates
                    coords = np.array(line.coords)
                    # Map lines are in world coordinates, only apply rotation
                    new_coords = self.transform_world_to_tier4(coords)
                    transformed_lines.append(LineString(new_coords))
                else:
                    # Transform raw polyline (world coordinates, only rotate)
                    transformed_lines.append(self.transform_world_to_tier4(line))
            info['gt_lines_instance'] = transformed_lines
        
        if 'gt_polylines' in info:
            for key in info['gt_polylines']:
                if isinstance(info['gt_polylines'][key], list):
                    transformed_polylines = []
                    for polyline in info['gt_polylines'][key]:
                        # Map polylines are in world coordinates, only rotate
                        transformed_polylines.append(self.transform_world_to_tier4(polyline))
                    info['gt_polylines'][key] = transformed_polylines
        
        return info
    
    def get_ann_info(self, index):
        """Get annotation info and transform to Tier4 coordinates.

        Overrides parent method to ensure GT bounding boxes and attr_labels
        are in Tier4 coordinate system for correct collision evaluation.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information with Tier4 transformations applied.
        """
        # Get annotations from parent class
        anns_results = super().get_ann_info(index)

        # Transform gt_bboxes_3d to Tier4 coordinates
        gt_bboxes_3d = anns_results['gt_bboxes_3d']
        if gt_bboxes_3d is not None and len(gt_bboxes_3d) > 0:
            # Extract box data
            box_tensor = gt_bboxes_3d.tensor.numpy()

            # Transform centers (WITH translation for absolute positions)
            centers = box_tensor[:, :3]
            centers_transformed = np.array([
                self.transform_position_to_tier4(center, apply_translation=True)
                for center in centers
            ])

            # Keep size unchanged (object-relative)
            sizes = box_tensor[:, 3:6]

            # Transform yaw angles
            yaws = box_tensor[:, 6]
            yaws_transformed = np.array([
                self.transform_yaw_to_tier4(yaw) for yaw in yaws
            ])

            # Transform velocities if present (rotation only, no translation)
            if box_tensor.shape[1] >= 9:
                velocities = box_tensor[:, 7:9]
                velocities_transformed = np.array([
                    self.transform_velocity_to_tier4(vel) for vel in velocities
                ])

                # Reconstruct box tensor with transformed values
                new_box_tensor = np.concatenate([
                    centers_transformed,
                    sizes,
                    yaws_transformed.reshape(-1, 1),
                    velocities_transformed
                ], axis=1)
            else:
                # No velocity
                new_box_tensor = np.concatenate([
                    centers_transformed,
                    sizes,
                    yaws_transformed.reshape(-1, 1)
                ], axis=1)

            # Create new LiDARInstance3DBoxes with transformed data
            from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
            gt_bboxes_3d_new = LiDARInstance3DBoxes(
                new_box_tensor,
                box_dim=new_box_tensor.shape[-1],
                origin=(0.5, 0.5, 0.5)
            ).convert_to(self.box_mode_3d)

            anns_results['gt_bboxes_3d'] = gt_bboxes_3d_new

        # Transform attr_labels
        if 'attr_labels' in anns_results and anns_results['attr_labels'] is not None:
            attr_labels = anns_results['attr_labels'].copy()

            if len(attr_labels) > 0:
                # attr_labels structure (for future_frames=6):
                # [0:12]: future_track_offset (6*2) - frame-to-frame displacements
                # [12:18]: future_mask (6) - unchanged
                # [18:19]: gt_fut_goal (1) - unchanged
                # [19:28]: agent_lcf_feat (9) breakdown:
                #   [19:21]: agent center position (2) - absolute position
                #   [21:22]: agent yaw (1)
                #   [22:24]: agent velocity (2)
                #   [24:27]: agent size w,l,h (3) - unchanged
                #   [27:28]: class index (1) - unchanged
                # [28:34]: future_yaw_offset (6) - relative angles, unchanged

                frames = self.future_frames

                # Transform future_track_offset (rotation only, no translation)
                track_offset_end = frames * 2
                for i in range(len(attr_labels)):
                    # Reshape to (frames, 2) for transformation
                    offsets = attr_labels[i, :track_offset_end].reshape(frames, 2)
                    offsets_transformed = np.array([
                        self.transform_position_to_tier4(offset, apply_translation=False)
                        for offset in offsets
                    ])
                    attr_labels[i, :track_offset_end] = offsets_transformed.reshape(-1)

                # Transform agent_lcf_feat
                lcf_start = frames * 2 + frames + 1  # Skip masks and goal
                for i in range(len(attr_labels)):
                    # Transform agent position (WITH translation)
                    agent_pos = attr_labels[i, lcf_start:lcf_start+2]
                    agent_pos_transformed = self.transform_position_to_tier4(
                        agent_pos, apply_translation=True
                    )
                    attr_labels[i, lcf_start:lcf_start+2] = agent_pos_transformed

                    # Transform agent yaw
                    agent_yaw = attr_labels[i, lcf_start+2]
                    agent_yaw_transformed = self.transform_yaw_to_tier4(agent_yaw)
                    attr_labels[i, lcf_start+2] = agent_yaw_transformed

                    # Transform agent velocity (rotation only)
                    agent_vel = attr_labels[i, lcf_start+3:lcf_start+5]
                    agent_vel_transformed = self.transform_velocity_to_tier4(agent_vel)
                    attr_labels[i, lcf_start+3:lcf_start+5] = agent_vel_transformed

                    # Size (w,l,h) at lcf_start+5:lcf_start+8 stays unchanged
                    # Class index at lcf_start+8 stays unchanged

                # future_yaw_offset at the end stays unchanged (relative angles)

                anns_results['attr_labels'] = attr_labels

        return anns_results

    def evaluate(self, results, *args, **kwargs):
        """
        Evaluate with Tier4 coordinate system.

        Results are already in Tier4 coordinates, so evaluation proceeds normally.
        """
        return super().evaluate(results, *args, **kwargs)

