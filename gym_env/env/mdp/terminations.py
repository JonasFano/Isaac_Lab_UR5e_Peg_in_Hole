# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


def is_peg_inserted(
    env: ManagerBasedRLEnv,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_height: float = 0.05,
    xy_threshold: float = 0.0025,
    z_variability: float = 0.002,
) -> torch.Tensor:

    """ Check if peg is in the hole. Returns a binary tensor: 1 if success, 0 otherwise. """
    object: RigidObject | Articulation = env.scene[object_cfg.name]
    hole: RigidObject | Articulation = env.scene[hole_cfg.name]

    hole_pos_w = hole.data.root_pos_w.clone()        # shape: [N, 3]
    object_pos_w = object.data.root_pos_w.clone()    # shape: [N, 3]

    # Compute position delta
    delta = object_pos_w - hole_pos_w       # shape: [N, 3]

    # print("XY L2 Norm: ", torch.linalg.vector_norm(delta[:, :2], dim=1))

    # Condition 1: L2 norm in XY is less than predefined threshold
    xy_ok = torch.linalg.vector_norm(delta[:, :2], dim=1) <= xy_threshold

    # Condition 2: Z error is in predefined range
    z_ok = (delta[:, 2] >= object_height - z_variability) & (delta[:, 2] <= object_height + z_variability)

    # print("Height: ", delta[:, 2])

    # Combine both
    success = (xy_ok & z_ok).bool()  # Returns true if both true, else false

    return success  # shape: [N]