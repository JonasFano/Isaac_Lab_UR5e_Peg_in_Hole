import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import compute_pose_error
import math 

if TYPE_CHECKING:
    from .impedance_control_cfg import ImpedanceControllerCfg


class ImpedanceController:
    def __init__(self, cfg: "ImpedanceControllerCfg", num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        self.target_dim = 3
        self.desired_ee_pose_b = torch.zeros((num_envs, 7), device=device)

        self.Kp = torch.tensor(cfg.stiffness, device=self.device).view(1, 6).repeat(self.num_envs, 1)
        if cfg.damping is None:
            self.Kd = 6.0 * 2.0 * torch.sqrt(self.Kp)
            self.Kd[:, 2] = 8.0 * 2.0 * torch.sqrt(self.Kp[:, 2])
        else:
            self.Kd = torch.tensor(cfg.damping, device=self.device).view(1, 6).repeat(self.num_envs, 1)

        self.max_torque_clamp = (
            torch.tensor(cfg.max_torque_clamping, device=device).view(1, -1).repeat(num_envs, 1)
            if cfg.max_torque_clamping is not None else None
        )



    @property
    def action_dim(self) -> int:
        return self.target_dim

    def reset(self):
        self.desired_ee_pos_b.zero_()

    def set_command(self, command: torch.Tensor, current_pos_b: torch.Tensor):
        """Set desired EE pose using a relative 3D position command and fixed orientation."""
        self.desired_ee_pose_b[:, :3] = current_pos_b + command
        # Fixed orientation (w, x, y, z)
        self.desired_ee_pose_b[:, 3:] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)


    def compute(
        self,
        jacobian_b: torch.Tensor,              # (num_envs, 6, num_dof)
        ee_pos_b: torch.Tensor,                # (num_envs, 3)
        ee_quat_b: torch.Tensor,               # (num_envs, 4)
        ee_linvel_b: torch.Tensor,             # (num_envs, 3)
        ee_angvel_b: torch.Tensor,             # (num_envs, 3)
        mass_matrix: torch.Tensor | None = None,
        gravity_b: torch.Tensor | None = None,
        coriolis_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_envs, _, num_dof = jacobian_b.shape
        dof_torques = torch.zeros((num_envs, num_dof), device=self.device)

        # Compute pose error
        pose_error_b = torch.cat(
            compute_pose_error(
                ee_pos_b,
                ee_quat_b,
                self.desired_ee_pose_b[:, :3],
                self.desired_ee_pose_b[:, 3:],
                rot_error_type="axis_angle",
            ),
            dim=-1,
        )

        vel_err_b = -ee_linvel_b # zero velocity is desired
        angvel_err_b = -ee_angvel_b

        # Compute desired 6D acceleration
        acc_des = self.Kp * pose_error_b + self.Kd * torch.cat([vel_err_b, angvel_err_b], dim=-1)


        # Compute torques
        if self.cfg.inertial_dynamics_decoupling:
            J_T = jacobian_b.transpose(1, 2)
            M_inv = torch.linalg.inv(mass_matrix)
            M_task_inv = torch.bmm(torch.bmm(jacobian_b, M_inv), J_T)
            M_task = torch.linalg.inv(M_task_inv)
            task_wrench = torch.bmm(M_task, acc_des.unsqueeze(-1)).squeeze(-1)
        else:
            task_wrench = acc_des

        dof_torques += torch.bmm(jacobian_b.transpose(1, 2), task_wrench.unsqueeze(-1)).squeeze(-1)

        if self.cfg.gravity_compensation:
            if gravity_b is None:
                raise ValueError("Gravity compensation requested but gravity_b is None.")
            dof_torques += gravity_b

        if self.cfg.coriolis_centrifugal_compensation:
            if coriolis_b is None:
                raise ValueError("Coriolis compensation requested but coriolis_b is None.")
            dof_torques += coriolis_b

        if self.max_torque_clamp is not None:
            dof_torques = torch.clamp(dof_torques, min=-self.max_torque_clamp, max=self.max_torque_clamp)

        return dof_torques