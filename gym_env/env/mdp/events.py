from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import carb
import math
import omni.physics.tensors.impl.api as physx
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_apply, matrix_from_quat
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_friction_coefficients(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    static_friction_distribution_params: tuple[float, float],
    dynamic_friction_distribution_params: tuple[float, float],
    restitution_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    make_consistent: bool = False,
    ):
    """
    Randomizes friction and restitution coefficients for specific asset bodies.

    Uses `_randomize_prop_by_op` to apply the selected randomization strategy.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if not isinstance(asset, (RigidObject, Articulation)):
        raise ValueError(
            f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{asset_cfg.name}'"
            f" with type: '{type(asset)}'."
        )

    # resolve environment ids
    env_ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()

    # print(env_ids)

    if isinstance(asset, Articulation) and asset_cfg.body_ids != slice(None):
        num_shapes_per_body = []
        for link_path in asset.root_physx_view.link_paths[0]:
            link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        
        # ensure the parsing is correct
        num_shapes = sum(num_shapes_per_body)
        expected_shapes = asset.root_physx_view.max_shapes
        if num_shapes != expected_shapes:
            raise ValueError(
                "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
            )
    else:
        # in this case, we don't need to do special indexing
        num_shapes_per_body = None

    # print(num_shapes_per_body)

    # Define the friction parameters and randomize them
    friction_params = {
        "static": static_friction_distribution_params,
        "dynamic": dynamic_friction_distribution_params,
        "restitution": restitution_distribution_params
    }

    # Get existing material properties
    materials = asset.root_physx_view.get_material_properties()
    material_samples = materials.clone()

    # print(materials)

    # Apply randomization
    for i, key in enumerate(["static", "dynamic", "restitution"]):
        material_samples[:, :, i] = _randomize_prop_by_op(
            material_samples[:, :, i],
            friction_params[key],
            env_ids,
            slice(None),
            operation=operation,
            distribution=distribution
        )

    # print(material_samples)
    # print(env_ids.shape)
    # print(env_ids.shape[0])

    # Ensure dynamic friction <= static friction if required
    if make_consistent:
        material_samples[:, :, 1] = torch.min(material_samples[:, :, 0], material_samples[:, :, 1])

    # update material buffer with new samples
    if num_shapes_per_body is not None:
        # sample material properties from the given ranges
        for body_id in asset_cfg.body_ids:
            # obtain indices of shapes for the body
            start_idx = sum(num_shapes_per_body[:body_id])
            end_idx = start_idx + num_shapes_per_body[body_id]

            # print(f"materials shape: {materials.shape}")
            # print(f"materials[env_ids, start_idx:end_idx] shape: {materials[env_ids, start_idx:end_idx].shape}")
            # print(f"material_samples[:, start_idx:end_idx] shape: {material_samples[:, start_idx:end_idx].shape}")

            # assign the new materials
            # material samples are of shape: num_env_ids x total_num_shapes x 3
            materials[env_ids, start_idx:end_idx] = material_samples[env_ids, start_idx:end_idx]
    elif num_shapes_per_body is None and env_ids.shape[0] != env.scene.num_envs:
        materials[env_ids] = material_samples[env_ids].unsqueeze(0)
    else:
        # assign all the materials
        materials[env_ids] = material_samples[:]

    # print(materials)

    # Apply new material properties to simulation
    asset.root_physx_view.set_material_properties(materials, env_ids)



def randomize_actuator_gains_custom(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation_stiffness: Literal["add", "scale", "abs"] = "abs",
    operation_damping: Literal["add", "scale", "abs"] = "abs",
    distribution_stiffness: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    distribution_damping: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
    """
    Randomize the actuator gains (stiffness & damping) for an articulation.

    This function applies randomization using `_randomize_prop_by_op` directly for:
    - Stiffness (`operation_stiffness`, `distribution_stiffness`)
    - Damping (`operation_damping`, `distribution_damping`)

    If a property distribution is not provided, it remains unchanged.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        Use it only during initialization to avoid performance overhead.
    """
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # Loop through actuators and randomize gains
    for actuator in asset.actuators.values():
        # Determine joint indices to randomize
        if isinstance(asset_cfg.joint_ids, slice):
            # we take all the joints of the actuator
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            else:
                global_indices = torch.tensor(actuator.joint_indices, device=asset.device)
        elif isinstance(actuator.joint_indices, slice):
            # we take the joints defined in the asset config
            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            # we take the intersection of the actuator joints and the asset config joints
            actuator_joint_indices = torch.tensor(actuator.joint_indices, device=asset.device)
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)

            # the indices of the joints in the actuator that have to be randomized
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue

            # maps actuator indices that have to be randomized to global joint indices
            global_indices = actuator_joint_indices[actuator_indices]

        # Randomize stiffness
        if stiffness_distribution_params is not None:
            stiffness = actuator.stiffness[env_ids].clone()
            stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
            stiffness = _randomize_prop_by_op(
                stiffness, 
                stiffness_distribution_params, 
                dim_0_ids=None, 
                dim_1_ids=actuator_indices, 
                operation=operation_stiffness, 
                distribution=distribution_stiffness
            )
            actuator.stiffness[env_ids] = stiffness
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)

        # Randomize damping
        if damping_distribution_params is not None:
            damping = actuator.damping[env_ids].clone()
            damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
            damping = _randomize_prop_by_op(
                damping, 
                damping_distribution_params, 
                dim_0_ids=None, 
                dim_1_ids=actuator_indices, 
                operation=operation_damping, 
                distribution=distribution_damping
            )
            actuator.damping[env_ids] = damping
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)




def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]

    # print(dim_0_ids)

    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data



def randomize_initial_state(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        tcp_rand_range_x: tuple[float, float] = (-0.0125, 0.0125),
        tcp_rand_range_y: tuple[float, float] = (-0.0125, 0.0125),
        tcp_rand_range_z: tuple[float, float] = (0.1, 0.125),
        tcp_rand_range_roll: tuple[float, float] = (0.0, 0.0),
        tcp_rand_range_pitch: tuple[float, float] = (math.pi, math.pi),
        tcp_rand_range_yaw: tuple[float, float] = (-3.14, 3.14),
        joint_names: list[str] = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        ee_body_name: str = "wrist_3_link",
        tcp_offset: list[float] = [0.0, 0.0, 0.15],
        ik_max_iters: int = 10,
        pos_error_threshold: float = 1e-3,
        angle_error_threshold: float = 1e-3,
        levenberg_marquardt_lambda: float = 0.01,
        default_joint_pos = [2.5, -2.0, 2.0, -1.5, -1.5, 0.0, 0.0, 0.0],
        gravity: list[float] = [0.0, 0.0, -9.81],
        object_rand_range_x: tuple[float, float] = (-0.005, 0.005),  #(-0.01, 0.01),
        object_rand_range_z: tuple[float, float] = (0.005, 0.025), #(0.01, 0.02),
        gripper_joint_names: list[str] = ["joint_left", "joint_right"],
        gripper_joint_pos_close: list[float] = [-0.025, -0.025],
        object_width: float = 0.008,
    ):
    """Randomize the starting TCP pose within a predefined range above the hole and randomize the peg pose and ensure it is grasped upon reset."""
    # Disable gravity.
    physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

    robot: RigidObject | Articulation = env.scene[robot_cfg.name]
    object: RigidObject | Articulation = env.scene[object_cfg.name]
    hole: RigidObject | Articulation = env.scene[hole_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    tcp_target_pose_b = torch.zeros(env.num_envs, 7, device=env.device)
    bad_envs = env_ids.clone()

    ik_attempt = 0

    # print("Randomize Start")

    while True:
        n_bad = bad_envs.shape[0]

        # print("Bad envs: ", n_bad)

        # sample only for bad envs!
        r = torch.empty(n_bad, device=env.device)
        tcp_target_pose_b[bad_envs, 0] = r.uniform_(*tcp_rand_range_x)
        tcp_target_pose_b[bad_envs, 1] = r.uniform_(*tcp_rand_range_y)
        tcp_target_pose_b[bad_envs, 2] = r.uniform_(*tcp_rand_range_z)
        
        # Orientation
        euler_angles = torch.zeros_like(tcp_target_pose_b[bad_envs, :3])
        euler_angles[:, 0].uniform_(*tcp_rand_range_roll)
        euler_angles[:, 1].uniform_(*tcp_rand_range_pitch)
        euler_angles[:, 2].uniform_(*tcp_rand_range_yaw)
        tcp_target_pose_b[bad_envs, 3:] = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

        hole_position_b = hole_position_in_robot_base_frame(robot, hole)[:, :3]
        tcp_target_pose_b[bad_envs, :3] += hole_position_b[bad_envs, :3]

        print("Hole Position: ", hole_position_b[env_ids, :])
        print("Sampled Pose: ", tcp_target_pose_b[env_ids, :])

        joint_ids, joint_names = robot.find_joints(joint_names)
        num_joints = len(joint_ids)
        body_ids, _ = robot.find_bodies(ee_body_name)
        body_idx = body_ids[0]

        # Avoid indexing across all joints for efficiency
        if num_joints == robot.num_joints:
            joint_ids = slice(None)

        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)

        body_offset = OffsetCfg(pos=tcp_offset)

        # Run IK only on bad_envs
        compute_inverse_kinematics(
            env, 
            env_ids=bad_envs, 
            robot=robot, 
            joint_ids=joint_ids, 
            body_idx=body_idx, 
            command=tcp_target_pose_b, 
            body_offset=body_offset,
            ik_max_iters=ik_max_iters,
            pos_error_threshold=pos_error_threshold,
            angle_error_threshold=angle_error_threshold,
            levenberg_marquardt_lambda=levenberg_marquardt_lambda
        )

        # Compute TCP pose error for all env_ids
        tcp_pose_error, _ = get_tcp_pose_error(env, env_ids, robot, body_idx, tcp_target_pose_b, body_offset)

        print("TCP Pose Error: ", tcp_pose_error[env_ids, :])

        # Determine new bad_envs
        pos_error = torch.linalg.norm(tcp_pose_error[:, :3], dim=1) > pos_error_threshold
        angle_error = torch.norm(tcp_pose_error[:, 3:7], dim=1) > angle_error_threshold
        any_error = torch.logical_or(pos_error, angle_error)

        print("Any Error: ", any_error)

        if env_ids.numel() == any_error.numel():
            bad_envs = env_ids[any_error]
        elif env_ids.numel() == 1 and any_error[env_ids.item()]:
            bad_envs = env_ids
        else:
            bad_envs = torch.empty(0, dtype=torch.long, device=env.device)

        # print(len(bad_envs))

        if len(bad_envs) == 0:
            break

        ik_attempt += 1

        if ik_attempt > 20:
            print(ik_attempt)

        # Set robot to default joint position in all bad_envs
        set_robot_to_default_joint_pos(env, robot, joints=default_joint_pos, env_ids=bad_envs, gripper_width = 0)

    # Get current end-effector pose in world base frame
    tcp_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    tcp_quat_w = ee_frame.data.target_quat_w[env_ids, 0, :]

    # Sample z offset for the object/peg in world frame
    object_sampled_z = torch.rand(len(env_ids), 1, device=tcp_pos_w.device) * (object_rand_range_z[1] - object_rand_range_z[0]) + object_rand_range_z[0]

    # Sample x offset in TCP frame
    object_sampled_x_tcp = torch.rand(len(env_ids), 1, device=tcp_pos_w.device) * (object_rand_range_x[1] - object_rand_range_x[0]) + object_rand_range_x[0]

    # Convert quaternions to rotation matrices
    tcp_rot_w = matrix_from_quat(tcp_quat_w)

    # Create local offset [x, 0, 0] in TCP frame
    object_offset_tcp = torch.cat([object_sampled_x_tcp, torch.zeros_like(object_sampled_x_tcp), torch.zeros_like(object_sampled_x_tcp)], dim=-1).unsqueeze(-1)  # [B, 3, 1]

    # Transform local offset into world frame
    object_x_offset_w = torch.bmm(tcp_rot_w, object_offset_tcp).squeeze(-1)  # [B, 3]

    # To test terminating environments if the peg is inserted in the hole
    # positions = hole.data.root_pos_w[env_ids, :3] 
    # positions[:, 2:3] += sampled_z + torch.tensor([0.05], device=env.device).repeat(len(env_ids), 1)
    # Otherwise use this
    new_object_pos = tcp_pos_w.clone() + object_x_offset_w
    new_object_pos[:, 2:3] += object_sampled_z
    new_object_quat = tcp_quat_w

    # set into the physics simulation
    object.write_root_pose_to_sim(torch.cat([new_object_pos, new_object_quat], dim=-1), env_ids=env_ids)
    object.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
    object.reset()

    # Close gripper
    # Set gripper_offset = 0.025 to fully open the gripper
    gripper_joint_pos_offset = (object_width - 0.001) / 2 # or 0.0005 
    adjusted_gripper_joint_pos_close = [v + gripper_joint_pos_offset for v in gripper_joint_pos_close]

    joint_pos = robot.data.joint_pos[env_ids, :].clone()

    gripper_joint_ids, _ = robot.find_joints(gripper_joint_names)
    if isinstance(gripper_joint_ids, list):
        gripper_joint_ids = torch.tensor(gripper_joint_ids, device=env.device)

    # joint_pos[:, gripper_joint_ids] = torch.tensor(gripper_joint_pos_close, device=env.device).repeat(len(env_ids), 1) # Fully closing the gripper
    joint_pos[:, gripper_joint_ids] = torch.tensor(adjusted_gripper_joint_pos_close, device=env.device).repeat(len(env_ids), 1) # Otherwise

    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    # Enable gravity
    physics_sim_view.set_gravity(carb.Float3(*gravity))

    # print("Randomize Finish")


# ---------------- Helper Functions ---------------------

def get_tcp_pose_error(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    robot: RigidObject | Articulation, 
    body_idx: int, 
    command: torch.Tensor,
    body_offset: OffsetCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    ee_pos_des = command[env_ids, 0:3]
    ee_quat_des = command[env_ids, 3:7]

    ee_pos_curr, ee_quat_curr = compute_frame_pose(env, robot, body_idx, body_offset, env_ids=env_ids)

    position_error, axis_angle_error = compute_pose_error(
        ee_pos_curr, ee_quat_curr, ee_pos_des, ee_quat_des, rot_error_type="axis_angle"
    )
    tcp_pose_error = torch.cat((position_error, axis_angle_error), dim=1)
    return tcp_pose_error, ee_quat_curr



def compute_inverse_kinematics(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor | None, 
    robot: RigidObject | Articulation, 
    joint_ids: torch.Tensor, 
    body_idx: int, 
    command: torch.Tensor, 
    body_offset: OffsetCfg,
    ik_max_iters: int = 10,
    pos_error_threshold: float = 1e-3,
    angle_error_threshold: float = 1e-3,
    levenberg_marquardt_lambda: float = 0.01
) -> torch.Tensor:
    """
    Solves inverse kinematics via iterative updates using the Jacobian.

    Args:
        env: The environment.
        env_ids: Environment indices to solve for.
        robot: Robot articulation.
        joint_ids: Joint indices to optimise.
        body_idx: Index of the end-effector body.
        command: Target poses [N, 7] (pos + quat).
        body_offset: Offset of the TCP relative to the end-effector link.
        ik_max_iters: Max number of IK iterations per call.
        pos_error_threshold: Early exit if all envs have position error below this.
        angle_error_threshold: Early exit if all envs have orientation error below this.

    Returns:
        Final pose error after last iteration [N, 7].
    """
    for _ in range(ik_max_iters):
        joint_pos = robot.data.joint_pos[env_ids].clone()

        tcp_pose_error, ee_quat_curr = get_tcp_pose_error(env, env_ids, robot, body_idx, command, body_offset)

        # Early stopping
        pos_error = torch.linalg.norm(tcp_pose_error[:, :3], dim=-1)
        ang_error = torch.norm(tcp_pose_error[:, 3:], dim=-1)
        if torch.all(pos_error < pos_error_threshold) and torch.all(ang_error < angle_error_threshold):
            break

        if torch.any(ee_quat_curr.norm(dim=-1) == 0):
            continue

        jacobian = compute_frame_jacobian(env, robot, env_ids, body_idx, joint_ids, body_offset)
        delta_joint_pos = compute_delta_joint_pos(env, tcp_pose_error, jacobian, levenberg_marquardt_lambda = levenberg_marquardt_lambda)

        joint_pos[:, joint_ids] += delta_joint_pos

        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)


def compute_delta_joint_pos(
    env: ManagerBasedRLEnv, 
    delta_pose: torch.Tensor, 
    jacobian: torch.Tensor, 
    levenberg_marquardt_lambda: float = 0.01,
) -> torch.Tensor:
    """Computes the change in joint position that yields the desired change in pose.

    The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
    to compute the delta-change in the joint-space that moves the robot closer to a desired
    end-effector position.

    Args:
        delta_pose: The desired delta pose in shape (N, 3) or (N, 6).
        jacobian: The geometric jacobian matrix in shape (N, 3, num_joints) or (N, 6, num_joints).

    Returns:
        The desired delta in joint space. Shape is (N, num-jointsß).
    """
    # compute the delta in joint-space
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    lambda_matrix = (levenberg_marquardt_lambda**2) * torch.eye(n=jacobian.shape[1], device=env.device)
    delta_joint_pos = (
        jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
    )
    delta_joint_pos = delta_joint_pos.squeeze(-1)

    return delta_joint_pos


def compute_frame_pose(
    env: ManagerBasedRLEnv,
    robot: RigidObject | Articulation,
    body_idx: int,
    body_offset: OffsetCfg,
    env_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    env_ids = slice(None) if env_ids is None else env_ids

    if body_offset is not None:
        offset_pos = torch.tensor(body_offset.pos, device=env.device).repeat(len(env_ids), 1)
        offset_rot = torch.tensor(body_offset.rot, device=env.device).repeat(len(env_ids), 1)

    ee_pos_w = robot.data.body_pos_w[env_ids, body_idx]
    ee_quat_w = robot.data.body_quat_w[env_ids, body_idx]
    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = robot.data.root_quat_w[env_ids]

    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    if body_offset is not None:
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(ee_pose_b, ee_quat_b, offset_pos, offset_rot)

    return ee_pose_b, ee_quat_b


def compute_frame_jacobian(
    env: ManagerBasedRLEnv,
    robot: RigidObject | Articulation,
    env_ids: torch.Tensor,
    body_idx: int,
    joint_ids: torch.Tensor,
    body_offset: OffsetCfg,
) -> torch.Tensor:
    """Computes the geometric Jacobian of the target frame in the root frame.

    This function accounts for the target frame offset and applies the necessary transformations to obtain
    the right Jacobian from the parent body Jacobian.
    """
    jacobian = jacobian_b(robot, body_idx, joint_ids)[env_ids]

    if body_offset is not None:
        offset_pos = torch.tensor(body_offset.pos, device=env.device).repeat(len(env_ids), 1)
        offset_rot = torch.tensor(body_offset.rot, device=env.device).repeat(len(env_ids), 1)

        # Modify the jacobian to account for the offset
        # -- translational part
        # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
        #        = (v_J_ee + w_J_ee x r_link_ee ) * q
        #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_pos), jacobian[:, 3:, :])
        # -- rotational part
        # w_link = R_link_ee @ w_ee
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(offset_rot), jacobian[:, 3:, :])

    return jacobian



def jacobian_b(robot: RigidObject | Articulation, body_idx: torch.Tensor, joint_ids: torch.Tensor) -> torch.Tensor:
    if robot.is_fixed_base:
        jacobi_body_idx = body_idx - 1
        jacobi_joint_ids = joint_ids
    else:
        jacobi_body_idx = body_idx
        jacobi_joint_ids = [i + 6 for i in joint_ids]

    jacobian = jacobian_w(robot, jacobi_body_idx, jacobi_joint_ids)
    base_rot = robot.data.root_quat_w
    base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
    jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
    return jacobian


def jacobian_w(robot: RigidObject | Articulation, jacobi_body_idx, jacobi_joint_ids) -> torch.Tensor:
    return robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids]


def set_robot_to_default_joint_pos(env: ManagerBasedRLEnv, robot: RigidObject | Articulation, joints, env_ids, gripper_width = 0):
    joint_pos = robot.data.default_joint_pos[env_ids]
    joint_pos[:, 6:] = gripper_width
    joint_pos[:, :8] = torch.tensor(joints, device=env.device)[None, :]
    joint_vel = torch.zeros_like(joint_pos)
    joint_effort = torch.zeros_like(joint_pos)

    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.reset()
    robot.set_joint_effort_target(joint_effort, env_ids=env_ids)


def hole_position_in_robot_base_frame(
    robot: RigidObject | Articulation,
    hole: RigidObject | Articulation,
) -> torch.Tensor:
    """Computes the hole pose in the robot's root frame."""
    hole_pose_w = hole.data.root_state_w[:, :7]

    # Transform the hole pose from the world frame to the robot's base frame
    hole_pos_b, hole_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], hole_pose_w[:, :3], hole_pose_w[:, 3:7],
    )
    hole_pose_b = torch.cat((hole_pos_b, hole_quat_b), dim=-1)
    return hole_pose_b
