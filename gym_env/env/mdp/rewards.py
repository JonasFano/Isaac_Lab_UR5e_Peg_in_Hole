# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_error_magnitude, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hole_ee_distance(
        env: ManagerBasedRLEnv,
        std: float,
        hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the hole using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    hole: RigidObject | Articulation = env.scene[hole_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Target hole position: (num_envs, 3)
    hole_pos_w = hole.data.root_pos_w

    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # Distance of the end-effector to the hole: (num_envs,)
    hole_ee_distance = torch.norm(hole_pos_w - ee_w, dim=1)

    # print("Hole_ee_distance: ", hole_ee_distance)

    return 1 - torch.tanh(hole_ee_distance / std)


def object_hole_orientation_error(
        env: ManagerBasedRLEnv, 
        hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (hole orientation) and the
    current orientation of the object (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset
    object: RigidObject | Articulation = env.scene[object_cfg.name]
    hole: RigidObject | Articulation = env.scene[hole_cfg.name]

    # obtain the desired and current orientations
    des_quat_w = hole.data.root_state_w[:, 3:7]
    curr_quat_w = object.data.root_state_w[:, 3:7]

    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    # print("Orientation error: ", error)
    return error


def action_rate_l2_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, :3] - env.action_manager.prev_action[:, :3]), dim=1)


def action_l2_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, :3]), dim=1)


def keypoint_distance(
        env: ManagerBasedRLEnv,
        hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_keypoints: int = 4,
        object_height: float = 0.05,
        a: float = 50,
        b: float = 2,
) -> torch.Tensor:
    """
    Reward based on keypoint distance between object and hole using keypoints from base to top.
    Object keypoints: from -object_height (base) to 0 (top).
    Hole keypoints: from 0 (base) to +object_height (top).
    """

    # Extract assets
    object: RigidObject | Articulation = env.scene[object_cfg.name]
    hole: RigidObject | Articulation = env.scene[hole_cfg.name]

    # Get root poses
    object_pos_w = object.data.root_pos_w.clone()      # (N, 3)
    object_quat_w = object.data.root_quat_w.clone()    # (N, 4)
    hole_pos_w = hole.data.root_pos_w.clone()
    hole_quat_w = hole.data.root_quat_w.clone()

    # Object keypoints: bottom to top (negative Z to 0)
    z_vals_object = torch.linspace(-object_height, 0.0, num_keypoints, device=env.device)
    keypoint_offsets_object = torch.zeros((num_keypoints, 3), device=env.device)
    keypoint_offsets_object[:, 2] = z_vals_object

    # Hole keypoints: bottom to top (0 to positive Z)
    z_vals_hole = torch.linspace(0.0, object_height, num_keypoints, device=env.device)
    keypoint_offsets_hole = torch.zeros((num_keypoints, 3), device=env.device)
    keypoint_offsets_hole[:, 2] = z_vals_hole

    object_quat_batch = object_quat_w.unsqueeze(1).repeat(1, num_keypoints, 1)     # (N, K, 4)
    hole_quat_batch = hole_quat_w.unsqueeze(1).repeat(1, num_keypoints, 1)   # (N, K, 4)

    # Batch for all envs
    kp_offsets_object_batch = keypoint_offsets_object.unsqueeze(0).repeat(env.num_envs, 1, 1)
    kp_offsets_hole_batch = keypoint_offsets_hole.unsqueeze(0).repeat(env.num_envs, 1, 1)

    # Transform to world frame
    object_keypoints_w = quat_apply(object_quat_batch, kp_offsets_object_batch) + object_pos_w.unsqueeze(1)
    hole_keypoints_w = quat_apply(hole_quat_batch, kp_offsets_hole_batch) + hole_pos_w.unsqueeze(1)

    # print(object_keypoints_w)
    # print(hole_keypoints_w)

    # Compute distances and average
    kp_distances = torch.norm(object_keypoints_w - hole_keypoints_w, dim=-1)  # (N, K)
    avg_kp_distance = kp_distances.mean(dim=1)

    # Reward function
    def squashing_fn(x, a, b):
        return 1 / (torch.exp(a * x) + b + torch.exp(-a * x)) # If x (mean keypoint distance) is very small, reward = 1 / (2 + b)

    reward = squashing_fn(avg_kp_distance, a, b)

    return reward


def is_peg_centered(
        env: ManagerBasedRLEnv,
        hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        object_height: float = 0.05,
        xy_threshold: float = 0.0025,
        z_threshold: float = 0.08,
        z_variability: float = 0.002,
) -> torch.Tensor:
    """Reward if peg is centered above the hole and is inserted below a Z height threshold."""
    
    # Extract assets
    object: RigidObject | Articulation = env.scene[object_cfg.name]
    hole: RigidObject | Articulation = env.scene[hole_cfg.name]

    # Get root positions
    object_pos_w = object.data.root_pos_w.clone()      # (N, 3)
    hole_pos_w = hole.data.root_pos_w.clone()

    # Compute 2D XY distance
    xy_dist = torch.linalg.vector_norm(hole_pos_w[:, 0:2] - object_pos_w[:, 0:2], dim=1)

    # print(xy_dist)

    # Check if XY distance is below threshold
    is_centered = xy_dist <= xy_threshold

    # Check if object is inserted below the Z threshold
    z_disp = object_pos_w[:, 2] - hole_pos_w[:, 2]
    is_low_enough = (z_disp <= z_threshold) & (z_disp >= object_height - z_variability) # If the peg is fully inserted, its height is sometimes 1 mm below its height

    # print(z_disp)

    # Return 1 if both conditions are met, else 0
    reward = torch.where(
        torch.logical_and(is_centered, is_low_enough),
        torch.ones_like(xy_dist, dtype=torch.float32),
        torch.zeros_like(xy_dist, dtype=torch.float32)
    )

    return reward


def penalize_contact_forces(
        env: ManagerBasedRLEnv,
    ) -> torch.Tensor:
    obs = env.observation_manager.compute_group("policy") # (Fx, Fy, Fz)
    forces = obs[:, 7:10]   # Fx, Fy, Fz
    return torch.norm(forces, dim=-1)


def penalize_contact_torque(
        env: ManagerBasedRLEnv,
    ) -> torch.Tensor:
    obs = env.observation_manager.compute_group("policy")
    torques = obs[:, 10:13] # Tx, Ty, Tz
    return torch.norm(torques, dim=-1)


def penalize_peg_missed_hole(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    termination_height: float = 0.07,     # ~2cm above table
    xy_margin: float = 0.0125,            # margin for contact-based failure
    xy_threshold: float = 0.02            # max allowed XY offset, independent of contact
) -> torch.Tensor:
    """
    Check if the peg either slipped off the hole area and touched the table,
    or is horizontally misaligned beyond a defined threshold.

    Returns a boolean tensor [N], where True indicates a missed hole.
    """
    object: RigidObject | Articulation = env.scene[object_cfg.name]
    hole: RigidObject | Articulation = env.scene[hole_cfg.name]

    hole_pos_w = hole.data.root_pos_w.clone()         # [N, 3]
    object_pos_w = object.data.root_pos_w.clone()     # [N, 3]

    delta_xy = object_pos_w[:, :2] - hole_pos_w[:, :2]  # [N, 2]

    # Condition 1: Z height too low AND XY offset too large
    z_missed = object_pos_w[:, 2] < termination_height  # [N]
    x_outside = torch.abs(delta_xy[:, 0]) > xy_margin
    y_outside = torch.abs(delta_xy[:, 1]) > xy_margin
    contact_missed = z_missed & (x_outside | y_outside)  # [N]

    # Condition 2: XY position exceeds general threshold (regardless of height)
    xy_offset = torch.norm(delta_xy, dim=-1)            # [N]
    xy_misaligned = xy_offset > xy_threshold            # [N]

    missed = contact_missed | xy_misaligned             # [N]
    return missed


