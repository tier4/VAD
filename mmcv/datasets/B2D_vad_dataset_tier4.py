"""
B2D VAD Dataset with Tier4 Coordinate System

This dataset applies the Tier4 coordinate transformation (Y->X, -X->Y) to B2D/CARLA data.
The coordinate origin is at the center of the rear axis, which is lower than typical LiDAR mount.
"""

import os
import copy
import numpy as np
import torch
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
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def get_axis_rotation_matrix():
        """Get the axis rotation matrix: Y->X, -X->Y, Z->Z"""
        return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    
    @staticmethod
    def transform_position_to_tier4(position):
        """
        Transform position from standard to Tier4 coordinates.
        Args:
            position: (N, 2) or (N, 3) or (2,) or (3,) array of positions
        Returns:
            Transformed positions in Tier4 coordinate system
        """
        axis_rotation = B2D_VAD_DatasetTier4.get_axis_rotation_matrix()
        
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
    
    @staticmethod
    def transform_velocity_to_tier4(velocity):
        """
        Transform velocity from standard to Tier4 coordinates.
        Args:
            velocity: (2,) or (N, 2) velocity [vx, vy]
        Returns:
            Transformed velocity in Tier4 coordinate system
        """
        axis_rotation = B2D_VAD_DatasetTier4.get_axis_rotation_matrix()
        
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
    
    @staticmethod
    def transform_yaw_to_tier4(yaw):
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
        Args:
            polyline: List of (x, y) points or numpy array
        Returns:
            Transformed polyline in Tier4 coordinate system
        """
        if polyline is None or len(polyline) == 0:
            return polyline
        
        polyline = np.array(polyline)
        return self.transform_position_to_tier4(polyline)
    
    def get_data_info(self, index):
        """
        Get data info with Tier4 coordinate transformation applied.
        
        Overrides parent method to apply coordinate transformations to:
        - Bounding boxes (center, velocity, yaw)
        - Ego trajectories (past and future)
        - Map polylines
        """
        # Get base info from parent class
        info = super().get_data_info(index)
        
        # Transform bounding boxes
        if 'gt_boxes' in info:
            for i in range(len(info['gt_boxes'])):
                info['gt_boxes'][i] = self.transform_bbox_to_tier4(info['gt_boxes'][i])
        
        if 'ann_infos' in info:
            for ann in info['ann_infos']:
                if 'translation' in ann:
                    pos = np.array(ann['translation'][:2])
                    new_pos = self.transform_position_to_tier4(pos)
                    ann['translation'][:2] = new_pos.tolist()
                
                if 'velocity' in ann:
                    vel = np.array(ann['velocity'][:2])
                    new_vel = self.transform_velocity_to_tier4(vel)
                    ann['velocity'][:2] = new_vel.tolist()
                
                if 'rotation' in ann:
                    yaw = quaternion_yaw(Quaternion(ann['rotation']))
                    new_yaw = self.transform_yaw_to_tier4(yaw)
                    ann['rotation'] = [np.cos(new_yaw/2), 0, 0, np.sin(new_yaw/2)]
        
        # Transform ego trajectories
        if 'ego_his_trajs' in info:
            info['ego_his_trajs'] = self.transform_trajectory_to_tier4(info['ego_his_trajs'])
        
        if 'ego_fut_trajs' in info:
            info['ego_fut_trajs'] = self.transform_trajectory_to_tier4(info['ego_fut_trajs'])
        
        if 'ego_his_pos' in info:
            info['ego_his_pos'] = self.transform_trajectory_to_tier4(info['ego_his_pos'])
        
        if 'ego_fut_pos' in info:
            info['ego_fut_pos'] = self.transform_trajectory_to_tier4(info['ego_fut_pos'])
        
        # Transform map polylines
        if 'gt_lines_instance' in info:
            transformed_lines = []
            for line in info['gt_lines_instance']:
                if isinstance(line, LineString):
                    # Transform LineString coordinates
                    coords = np.array(line.coords)
                    new_coords = self.transform_position_to_tier4(coords)
                    transformed_lines.append(LineString(new_coords))
                else:
                    # Transform raw polyline
                    transformed_lines.append(self.transform_polyline_to_tier4(line))
            info['gt_lines_instance'] = transformed_lines
        
        if 'gt_polylines' in info:
            for key in info['gt_polylines']:
                if isinstance(info['gt_polylines'][key], list):
                    transformed_polylines = []
                    for polyline in info['gt_polylines'][key]:
                        transformed_polylines.append(self.transform_polyline_to_tier4(polyline))
                    info['gt_polylines'][key] = transformed_polylines
        
        return info
    
    def evaluate(self, results, *args, **kwargs):
        """
        Evaluate with Tier4 coordinate system.
        
        Results are already in Tier4 coordinates, so evaluation proceeds normally.
        """
        return super().evaluate(results, *args, **kwargs)