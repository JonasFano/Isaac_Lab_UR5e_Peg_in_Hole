# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, matrix_from_quat
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def quat_rotate_vector(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector by a quaternion using provided utility functions.
    
    Args:
        quat: [..., 4] tensor representing quaternions (w, x, y, z).
        vec: [..., 3] tensor representing the vectors to rotate.
    
    Returns:
        Rotated vector of shape [..., 3].
    """
    # Ensure the quaternion is normalized to avoid unintended scaling effects
    quat = torch.nn.functional.normalize(quat, p=2, dim=-1)
    
    # Rotate the input vector using the quaternion
    rotated_vec = quat_apply(quat, vec)
    
    return rotated_vec


def get_current_tcp_pose(env: ManagerBasedRLEnv, gripper_offset: List[float], robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Compute the current TCP pose in both the base frame and world frame.
    
    Args:
        env: ManagerBasedRLEnv object containing the virtual environment.
        robot_cfg: Configuration for the robot entity, defaults to "robot".
    
    Returns:
        tcp_pose_b: TCP pose in the robot's base frame (position + quaternion).
    """
    # Access the robot object from the scene using the provided configuration
    robot: RigidObject | Articulation = env.scene[robot_cfg.name]

    # Clone the body states in the world frame to avoid modifying the original tensor
    body_state_w_list = robot.data.body_state_w.clone()

    # Extract the pose of the end-effector (position + orientation) in the world frame
    ee_pose_w = body_state_w_list[:, robot_cfg.body_ids[0], :7]

    # Define the offset from the end-effector frame to the TCP in the end-effector frame
    offset_ee = torch.tensor(gripper_offset, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(env.scene.num_envs, 1)

    # Rotate the offset from the end-effector frame to the world frame
    offset_w = quat_rotate_vector(ee_pose_w[:, 3:7], offset_ee)

    # Compute the TCP pose in the world frame by adding the offset to the end-effector's position
    tcp_pose_w = torch.cat((ee_pose_w[:, :3] + offset_w, ee_pose_w[:, 3:7]), dim=-1)

    # Transform the TCP pose from the world frame to the robot's base frame
    tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],  # Robot base position in world frame
        robot.data.root_state_w[:, 3:7],  # Robot base orientation in world frame
        tcp_pose_w[:, :3],  # TCP position in world frame
        tcp_pose_w[:, 3:7]  # TCP orientation in world frame
    )

    # # Convert orientation from quat to axis-angle
    # tcp_axis_angle_b = axis_angle_from_quat(tcp_quat_b)
    # # Concatenate the position and axis-angle into a single tensor
    # tcp_pose_b = torch.cat((tcp_pos_b, tcp_axis_angle_b), dim=-1)

    tcp_pose_b = torch.cat((tcp_pos_b, tcp_quat_b), dim=-1)
    return tcp_pose_b


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject | Articulation = env.scene[asset_cfg.name]
    object: RigidObject | Articulation = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    object_pose_b = torch.cat((object_pos_b, object_quat_b), dim=-1)
    return object_pose_b


def body_incoming_wrench_transform(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Incoming spatial wrench on bodies of an articulation in the simulation world frame.
    Converts from end-effector frame to world frame and vice versa.
    """
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Obtain the link incoming forces in an unknown frame
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force()[:, asset_cfg.body_ids]

    # Get the end-effector transformation (rotation matrix) in the world frame
    ee_transform = asset.data.body_state_w[:, asset_cfg.body_ids[-1], :7]  # Assuming last body is EE
    ee_quat = ee_transform[:, 3:7]  # Extract quaternion (w, x, y, z)

    # Convert quaternion to rotation matrix
    ee_rot_matrix = matrix_from_quat(ee_quat)  # Shape: (num_envs, 3, 3)

    # Define the rotation matrix for -90° around the y-axis (to EE frame, as determined empirically)
    R_y_neg_90 = torch.tensor([
        [0,  0, 1],
        [0,  -1,  0],
        [-1,  0,  0]
    ], dtype=torch.float32, device=link_incoming_forces.device)  # Ensure it matches tensor device

    # Extract force and torque components (assuming shape: (num_envs, num_links, 6))
    forces = link_incoming_forces[..., :3]  # First 3 components (forces)
    torques = link_incoming_forces[..., 3:]  # Last 3 components (torques)

    # print(forces)

    # Transform from world frame to EE frame
    forces_ee = torch.matmul(forces, R_y_neg_90.T)
    torques_ee = torch.matmul(torques, R_y_neg_90.T)

    # print("End-effector forces: ", forces_ee)

    # Transform from EE frame to world frame using end-effector rotation matrix
    forces_world = torch.matmul(forces_ee, ee_rot_matrix.transpose(1, 2))
    torques_world = torch.matmul(torques_ee, ee_rot_matrix.transpose(1, 2))

    # Concatenate transformed force and torque to return full wrench in world frame
    wrench_world = torch.cat((forces_world, torques_world), dim=-1)

    # Cap wrench values to +/- 10000
    wrench_world = torch.clamp(wrench_world, min=-5000.0, max=5000.0)

    # print("World forces: ", forces_world)

    # ---------- Logging to disk directly inside the function ----------
    try:
        log_path = Path(__file__).resolve().parents[3] / "data" / "wrench_log_minmax.bin"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute min and max over all environments
        min_wrench = torch.min(wrench_world, dim=0).values  # shape: (6,)
        max_wrench = torch.max(wrench_world, dim=0).values  # shape: (6,)
        wrench_minmax = torch.cat([min_wrench, max_wrench], dim=0)  # shape: (12,)

        # Write to binary file
        wrench_np = wrench_minmax.detach().cpu().numpy().astype(np.float32)
        with open(log_path, "ab") as f:
            wrench_np.tofile(f)
    except Exception as e:
        print(f"[Wrench Logging Error] {e}")
    # ------------------------------------------------------------------

    w = wrench_world.view(env.num_envs, -1)

    return w


def noisy_hole_pose_estimate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    noise_std: float = 0.0025,  # 2.5 mm
) -> torch.Tensor:
    """Provide a noisy estimate of the hole position in the robot's root frame."""
    robot: RigidObject | Articulation = env.scene[asset_cfg.name]
    hole: RigidObject | Articulation = env.scene[hole_cfg.name]

    # Get hole position in world frame
    hole_pos_w = hole.data.root_pos_w[:, :3]

    # Transform to robot base frame
    hole_pos_b, hole_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], hole_pos_w
    )

    # Add Gaussian noise to X and Y
    xy_noise = torch.randn_like(hole_pos_b[:, :2]) * noise_std
    hole_pos_b[:, :2] += xy_noise

    hole_pose_b = torch.cat((hole_pos_b, hole_quat_b), dim=-1)

    return hole_pose_b
