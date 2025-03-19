from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_friction_coefficients(
    env: ManagerBasedEnv,
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
    env: ManagerBasedEnv,
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
