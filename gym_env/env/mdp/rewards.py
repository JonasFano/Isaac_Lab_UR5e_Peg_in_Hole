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
from isaaclab.utils.math import quat_error_magnitude

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
    cube_pos_w = hole.data.root_pos_w

    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # Distance of the end-effector to the hole: (num_envs,)
    hole_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

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
    # extract the asset (to enable type hinting)
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