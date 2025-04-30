from typing import TYPE_CHECKING

import torch
from typing import Sequence
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.assets.articulation import Articulation
import isaaclab.utils.math as math_utils

from ...controller.impedance_control import ImpedanceController

if TYPE_CHECKING:
    from . import actions_cfg


class ImpedanceControllerAction(ActionTerm):
    """Action term using a custom impedance controller with position-only relative input."""

    cfg: "ImpedanceControllerActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "ImpedanceControllerActionCfg", env):
        super().__init__(cfg, env)

        # resolve joints
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_dof = len(self._joint_ids)
        if self._num_dof == self._asset.num_joints:
            self._joint_ids = slice(None)

        # resolve body
        body_ids, _ = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body name match for {cfg.body_name}")
        self._body_idx = body_ids[0]
        self._jacobi_joint_ids = self._joint_ids if self._asset.is_fixed_base else [i + 6 for i in self._joint_ids]
        self._jacobi_body_idx = self._body_idx - 1 if self._asset.is_fixed_base else self._body_idx

        # controller
        self._controller = ImpedanceController(cfg.controller_cfg, self.num_envs, self.device)

        # buffers
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._jacobian_b = torch.zeros((self.num_envs, 6, self._num_dof), device=self.device)
        self._mass_matrix = torch.zeros((self.num_envs, self._num_dof, self._num_dof), device=self.device)
        self._gravity = torch.zeros((self.num_envs, self._num_dof), device=self.device)
        self._coriolis = torch.zeros((self.num_envs, self._num_dof), device=self.device)

        self._ee_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._ee_linvel_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._ee_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self._ee_angvel_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._joint_efforts = torch.zeros((self.num_envs, self._num_dof), device=self.device)

        self._tcp_offset = (
            torch.tensor(cfg.tcp_offset, device=self.device).view(1, 3).expand(self.num_envs, 3)
            if cfg.tcp_offset else None
        )
        self._scale = torch.tensor(cfg.action_scale, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale

        self._update_ee_pose()
        self._controller.set_command(self._processed_actions, self._ee_pos_b)

    def apply_actions(self):
        self._update_dynamics()
        self._update_ee_pose()
        self._update_ee_velocity()
        self._update_ee_angvel()

        self._joint_efforts[:] = self._controller.compute(
            jacobian_b=self._jacobian_b,
            ee_pos_b=self._ee_pos_b,
            ee_quat_b=self._ee_quat_b,
            ee_linvel_b=self._ee_linvel_b,
            ee_angvel_b=self._ee_angvel_b,
            mass_matrix=self._mass_matrix,
            gravity_b=self._gravity,
            coriolis_b=self._coriolis,
        )

        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def _update_ee_pose(self):
        pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        root_pos = self._asset.data.root_pos_w
        root_quat = self._asset.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, pos_w, quat_w)

        if self._tcp_offset is not None:
            # rotate TCP offset from EEF frame into base frame
            tcp_offset_b = math_utils.quat_rotate(ee_quat_b, self._tcp_offset)
            self._ee_pos_b[:] = ee_pos_b + tcp_offset_b
        else:
            self._ee_pos_b[:] = ee_pos_b

        self._ee_quat_b[:] = ee_quat_b


    def _update_ee_velocity(self):
        vel_w = self._asset.data.body_vel_w[:, self._body_idx, 0:3]
        root_vel = self._asset.data.root_vel_w[:, 0:3]
        rel_vel = vel_w - root_vel
        self._ee_linvel_b[:] = math_utils.quat_rotate_inverse(self._asset.data.root_quat_w, rel_vel)

    def _update_ee_angvel(self):
        ang_vel_w = self._asset.data.body_vel_w[:, self._body_idx, 3:6]
        root_angvel_w = self._asset.data.root_vel_w[:, 3:6]
        rel_angvel_w = ang_vel_w - root_angvel_w
        self._ee_angvel_b[:] = math_utils.quat_rotate_inverse(self._asset.data.root_quat_w, rel_angvel_w)


    def _update_dynamics(self):
        self._jacobian_b[:] = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        base_rot = self._asset.data.root_quat_w
        rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        self._jacobian_b[:, :3] = torch.bmm(rot_matrix, self._jacobian_b[:, :3])
        self._jacobian_b[:, 3:] = torch.bmm(rot_matrix, self._jacobian_b[:, 3:])
        self._mass_matrix[:] = self._asset.root_physx_view.get_generalized_mass_matrices()[:, self._joint_ids, :][:, :, self._joint_ids]
        self._gravity[:] = self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]
        self._coriolis[:] = self._asset.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()[:, self._joint_ids]
