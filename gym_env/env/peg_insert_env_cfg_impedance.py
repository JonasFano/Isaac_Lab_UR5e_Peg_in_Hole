from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from . import mdp
import os
from gym_env.env.mdp.actions.actions_cfg import ImpedanceControllerActionCfg

from taskparameters_peginsert_impedance import TaskParams

##
# Scene definition
##

MODEL_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), "scene_models")

@configclass
class UR5e_PegInsertSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object."""
    # articulation
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(MODEL_PATH, "ur5e_robotiq_hand_e.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),  
        init_state=ArticulationCfg.InitialStateCfg(
            pos=TaskParams.robot_init_pos, 
            joint_pos={
                "shoulder_pan_joint": TaskParams.robot_initial_joint_pos[0], 
                "shoulder_lift_joint": TaskParams.robot_initial_joint_pos[1], 
                "elbow_joint": TaskParams.robot_initial_joint_pos[2], 
                "wrist_1_joint": TaskParams.robot_initial_joint_pos[3], 
                "wrist_2_joint": TaskParams.robot_initial_joint_pos[4], 
                "wrist_3_joint": TaskParams.robot_initial_joint_pos[5], 
                "joint_left": TaskParams.robot_initial_joint_pos[6],
                "joint_right": TaskParams.robot_initial_joint_pos[7],
            }
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # Match all joints
                velocity_limit={
                    "shoulder_pan_joint": TaskParams.robot_vel_limit,
                    "shoulder_lift_joint": TaskParams.robot_vel_limit,
                    "elbow_joint": TaskParams.robot_vel_limit,
                    "wrist_1_joint": TaskParams.robot_vel_limit,
                    "wrist_2_joint": TaskParams.robot_vel_limit,
                    "wrist_3_joint": TaskParams.robot_vel_limit,
                    "joint_left": TaskParams.gripper_vel_limit,
                    "joint_right": TaskParams.gripper_vel_limit,
                },
                effort_limit={
                    "shoulder_pan_joint": TaskParams.robot_effort_limit,
                    "shoulder_lift_joint": TaskParams.robot_effort_limit,
                    "elbow_joint": TaskParams.robot_effort_limit,
                    "wrist_1_joint": TaskParams.robot_effort_limit,
                    "wrist_2_joint": TaskParams.robot_effort_limit,
                    "wrist_3_joint": TaskParams.robot_effort_limit,
                    "joint_left": TaskParams.gripper_effort_limit,
                    "joint_right": TaskParams.gripper_effort_limit,
                },
                stiffness = {
                    "shoulder_pan_joint": TaskParams.robot_stiffness,
                    "shoulder_lift_joint": TaskParams.robot_stiffness,
                    "elbow_joint": TaskParams.robot_stiffness,
                    "wrist_1_joint": TaskParams.robot_stiffness,
                    "wrist_2_joint": TaskParams.robot_stiffness,
                    "wrist_3_joint": TaskParams.robot_stiffness,
                    "joint_left": TaskParams.gripper_stiffness,
                    "joint_right": TaskParams.gripper_stiffness,
                },
                damping = {
                    "shoulder_pan_joint": TaskParams.robot_damping,
                    "shoulder_lift_joint": TaskParams.robot_damping,
                    "elbow_joint": TaskParams.robot_damping,
                    "wrist_1_joint": TaskParams.robot_damping,
                    "wrist_2_joint": TaskParams.robot_damping,
                    "wrist_3_joint": TaskParams.robot_damping,
                    "joint_left": TaskParams.gripper_damping,
                    "joint_right": TaskParams.gripper_damping,
                }
            )
        }
    )

    # Add peg and hole objects on the table
    hole: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/hole",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_hole_8mm.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=TaskParams.hole_init_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=TaskParams.hole_init_pos, rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    object: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_peg_8mm.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=TaskParams.object_init_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=TaskParams.object_init_pos, rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.74]), 
        spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Lights/Dome", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )    
    
    # Add the siegmund table to the scene
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table", 
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(MODEL_PATH, "Single_Siegmund_table.usd")), 
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )



##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Set actions
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg | ImpedanceControllerActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # object_pos = ObsTerm(
        #     mdp.object_position_in_robot_root_frame,
        #     params={"asset_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object"),}
        # )

        # tcp_pose = ObsTerm(
        #     func=mdp.get_current_tcp_pose,
        #     params={"gripper_offset": TaskParams.gripper_offset, "robot_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
        #     # noise=Unoise(n_min=TaskParams.tcp_pose_unoise_min, n_max=TaskParams.tcp_pose_unoise_max),
        # )
        tcp_pose = ObsTerm(
            func=mdp.get_current_tcp_position,
            params={"gripper_offset": TaskParams.gripper_offset, "robot_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
            noise=Unoise(n_min=TaskParams.tcp_pose_unoise_min, n_max=TaskParams.tcp_pose_unoise_max),
        )

        ee_wrench_b = ObsTerm(
            func=mdp.body_incoming_wrench_transform,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ) # Small force/torque if not in contact, small but noticeable changes when moving, gets big when in contact

        # noisy_hole_pose_estimate = ObsTerm(
        #     func=mdp.noisy_hole_pose_estimate,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "hole_cfg": SceneEntityCfg("hole"),
        #         "noise_std": TaskParams.noise_std_hole_pose, 
        #     },
        # )
        noisy_hole_pose_estimate = ObsTerm(
            func=mdp.noisy_hole_pose_estimate_pos_only,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "hole_cfg": SceneEntityCfg("hole"),
                "noise_std": TaskParams.noise_std_hole_pose, 
            },
            noise=Unoise(n_min=TaskParams.tcp_pose_unoise_min, n_max=TaskParams.tcp_pose_unoise_max),
        )

        actions = ObsTerm(
            func=mdp.last_action
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomize the hole position 
    reset_hole_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": TaskParams.hole_randomize_pose_range_x, "y": TaskParams.hole_randomize_pose_range_y, "z": TaskParams.hole_randomize_pose_range_z}, # "yaw": TaskParams.hole_randomize_pose_range_yaw},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("hole"),
        },
    )

    randomize_impedance_gains = EventTerm(
        func=mdp.randomize_impedance_gains,
        mode="reset",
        params={
            "stiffness_ranges": TaskParams.impedance_stiffness_ranges,
            "damping_ratio_ranges": TaskParams.impedance_damping_ratio_ranges,
            "operation": TaskParams.impedance_gain_randomization_operation,
            "distribution": TaskParams.impedance_gain_randomization_distribution,
            "groupings": TaskParams.impedance_gain_groupings,
        }
    )

    # To test randomize_initial_state domain randomization
    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.2, -0.2), "y": (0.3, 0.5), "z": TaskParams.hole_randomize_pose_range_z}, # "yaw": TaskParams.hole_randomize_pose_range_yaw},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )

    specify_gripper_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, 
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="hand_base"),
            "mass_distribution_params": (1.07, 1.07),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        }
    )

    randomize_initial_robot_state = EventTerm(
        func=mdp.randomize_initial_state,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "hole_cfg": SceneEntityCfg("hole"),
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "tcp_rand_range_x": TaskParams.tcp_rand_range_x,
            "tcp_rand_range_y": TaskParams.tcp_rand_range_y,
            "tcp_rand_range_z": TaskParams.tcp_rand_range_z,
            "tcp_rand_range_roll": TaskParams.tcp_rand_range_roll,
            "tcp_rand_range_pitch": TaskParams.tcp_rand_range_pitch,
            "tcp_rand_range_yaw": TaskParams.tcp_rand_range_yaw,
            "joint_names": TaskParams.joint_names,
            "ee_body_name": TaskParams.ee_body_name,
            "tcp_offset": TaskParams.gripper_offset,
            "ik_max_iters": TaskParams.ik_max_iters,
            "pos_error_threshold": TaskParams.pos_error_threshold,
            "angle_error_threshold": TaskParams.angle_error_threshold,
            "levenberg_marquardt_lambda": TaskParams.levenberg_marquardt_lambda,
            "default_joint_pos": TaskParams.robot_initial_joint_pos,
            "gravity": TaskParams.gravity,
            "object_rand_range_x": TaskParams.object_rand_pos_range_x,
            "object_rand_range_z": TaskParams.object_rand_pos_range_z,
            "gripper_joint_names": TaskParams.gripper_joint_names,
            "gripper_joint_pos_close": TaskParams.gripper_joint_pos_close,
            "object_width": TaskParams.object_width,
        }
    )

    randomize_gripper_fingers_friction_coefficients = EventTerm(
        func=mdp.randomize_friction_coefficients,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=TaskParams.gripper_body_names),
            "static_friction_distribution_params": TaskParams.gripper_static_friction_distribution_params,
            "dynamic_friction_distribution_params": TaskParams.gripper_dynamic_friction_distribution_params,
            "restitution_distribution_params": TaskParams.gripper_restitution_distribution_params,
            "operation": TaskParams.gripper_randomize_friction_operation,
            "distribution": TaskParams.gripper_randomize_friction_distribution,
            "make_consistent": TaskParams.gripper_randomize_friction_make_consistent,  
        }
    )

    randomize_object_friction_coefficients = EventTerm(
        func=mdp.randomize_friction_coefficients,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names="forge_round_peg_8mm"),
            "static_friction_distribution_params": TaskParams.object_static_friction_distribution_params,
            "dynamic_friction_distribution_params": TaskParams.object_dynamic_friction_distribution_params,
            "restitution_distribution_params": TaskParams.object_restitution_distribution_params,
            "operation": TaskParams.object_randomize_friction_operation,
            "distribution": TaskParams.object_randomize_friction_distribution,
            "make_consistent": TaskParams.object_randomize_friction_make_consistent,  
        }
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    ######################################################
    # Dense keypoint distance rewards
    # keypoint_distance_coarse = RewTerm(
    #     func=mdp.keypoint_distance,
    #     params={
    #         "hole_cfg": SceneEntityCfg("hole"),
    #         "object_cfg": SceneEntityCfg("object"),
    #         "num_keypoints": TaskParams.num_keypoints,
    #         "object_height": TaskParams.object_height,
    #         "a": TaskParams.coarse_kernel_a,
    #         "b": TaskParams.coarse_kernel_b,
    #     },
    #     weight=TaskParams.keypoint_distance_coarse_weight,
    # )

    # keypoint_distance_fine = RewTerm(
    #     func=mdp.keypoint_distance,
    #     params={
    #         "hole_cfg": SceneEntityCfg("hole"),
    #         "object_cfg": SceneEntityCfg("object"),
    #         "num_keypoints": TaskParams.num_keypoints,
    #         "object_height": TaskParams.object_height,
    #         "a": TaskParams.fine_kernel_a,
    #         "b": TaskParams.fine_kernel_b,
    #     },
    #     weight=TaskParams.keypoint_distance_fine_weight,
    # )

    keypoint_distance_coarse = RewTerm(
        func=mdp.sequential_keypoint_distance,
        params={
            "hole_cfg": SceneEntityCfg("hole"),
            "object_cfg": SceneEntityCfg("object"),
            "num_keypoints": TaskParams.num_keypoints,
            "object_height": TaskParams.object_height,
            "keypoint_start_height": TaskParams.hole_height,
            "a": TaskParams.coarse_kernel_a,
            "b": TaskParams.coarse_kernel_b,
            "check_centered": False,
            "xy_threshold": TaskParams.is_peg_centered_xy_threshold,
            "z_threshold": TaskParams.is_peg_centered_z_threshold,
            "z_variability": TaskParams.is_peg_centered_z_variability,
        },
        weight=TaskParams.keypoint_distance_coarse_weight,
    )

    keypoint_distance_fine = RewTerm(
        func=mdp.sequential_keypoint_distance,
        params={
            "hole_cfg": SceneEntityCfg("hole"),
            "object_cfg": SceneEntityCfg("object"),
            "num_keypoints": TaskParams.num_keypoints,
            "object_height": TaskParams.object_height,
            "keypoint_start_height": TaskParams.hole_height,
            "a": TaskParams.fine_kernel_a,
            "b": TaskParams.fine_kernel_b,            
            "check_centered": False,
            "xy_threshold": TaskParams.is_peg_centered_xy_threshold,
            "z_threshold": TaskParams.is_peg_centered_z_threshold,
            "z_variability": TaskParams.is_peg_centered_z_variability,
        },
        weight=TaskParams.keypoint_distance_fine_weight,
    )

    keypoint_distance_inside_hole_coarse = RewTerm(
        func=mdp.sequential_keypoint_distance,
        params={
            "hole_cfg": SceneEntityCfg("hole"),
            "object_cfg": SceneEntityCfg("object"),
            "num_keypoints": TaskParams.num_keypoints,
            "object_height": TaskParams.object_height,
            "keypoint_start_height": 0.0025,
            "a": TaskParams.coarse_kernel_a,
            "b": TaskParams.coarse_kernel_b,
            "check_centered": True,
            "xy_threshold": TaskParams.is_peg_centered_xy_threshold,
            "z_threshold": TaskParams.is_peg_centered_z_threshold,
            "z_variability": TaskParams.is_peg_centered_z_variability,
        },
        weight=TaskParams.keypoint_distance_coarse_weight,
    )

    keypoint_distance_inside_hole_fine = RewTerm(
        func=mdp.sequential_keypoint_distance,
        params={
            "hole_cfg": SceneEntityCfg("hole"),
            "object_cfg": SceneEntityCfg("object"),
            "num_keypoints": TaskParams.num_keypoints,
            "object_height": TaskParams.object_height,
            "keypoint_start_height": 0.0025,
            "a": TaskParams.fine_kernel_a,
            "b": TaskParams.fine_kernel_b,
            "check_centered": True,
            "xy_threshold": TaskParams.is_peg_centered_xy_threshold,
            "z_threshold": TaskParams.is_peg_centered_z_threshold,
            "z_variability": TaskParams.is_peg_centered_z_variability,
        },
        weight=TaskParams.keypoint_distance_fine_weight,
    )
    ######################################################


    ######################################################
    # Sparse rewards
    is_peg_centered = RewTerm(
        func=mdp.is_peg_centered,
        params={
            "hole_cfg": SceneEntityCfg("hole"),
            "object_cfg": SceneEntityCfg("object"),
            "object_height": TaskParams.object_height,
            "xy_threshold": TaskParams.is_peg_centered_xy_threshold,
            "z_threshold": TaskParams.is_peg_centered_z_threshold,
            "z_variability": TaskParams.is_peg_centered_z_variability,
        },
        weight=TaskParams.is_peg_centered_weight,
    )

    # is_peg_centered_top = RewTerm(
    #     func=mdp.is_peg_centered,
    #     params={
    #         "hole_cfg": SceneEntityCfg("hole"),
    #         "object_cfg": SceneEntityCfg("object"),
    #         "object_height": TaskParams.object_height,
    #         "xy_threshold": TaskParams.is_peg_centered_xy_threshold,
    #         "z_threshold": TaskParams.is_peg_centered_z_threshold,
    #         "z_variability": TaskParams.is_peg_centered_z_variability_top,
    #     },
    #     weight=TaskParams.is_peg_centered_weight_top,
    # )

    # is_peg_centered_middle = RewTerm(
    #     func=mdp.is_peg_centered,
    #     params={
    #         "hole_cfg": SceneEntityCfg("hole"),
    #         "object_cfg": SceneEntityCfg("object"),
    #         "object_height": TaskParams.object_height,
    #         "xy_threshold": TaskParams.is_peg_centered_xy_threshold,
    #         "z_threshold": TaskParams.is_peg_centered_z_threshold,
    #         "z_variability": TaskParams.is_peg_centered_z_variability_middle,
    #     },
    #     weight=TaskParams.is_peg_centered_weight_middle,
    # )

    # is_peg_centered_bottom = RewTerm(
    #     func=mdp.is_peg_centered,
    #     params={
    #         "hole_cfg": SceneEntityCfg("hole"),
    #         "object_cfg": SceneEntityCfg("object"),
    #         "object_height": TaskParams.object_height,
    #         "xy_threshold": TaskParams.is_peg_centered_xy_threshold,
    #         "z_threshold": TaskParams.is_peg_centered_z_threshold,
    #         "z_variability": TaskParams.is_peg_centered_z_variability_bottom,
    #     },
    #     weight=TaskParams.is_peg_centered_weight_bottom,
    # )
    ######################################################

    is_peg_inserted = RewTerm(
        func=mdp.is_terminated_term,
        params={
            "term_keys": "is_peg_inserted", 
        },
        weight=TaskParams.is_peg_inserted_weight,
    )


    ######################################################
    peg_missed_hole = RewTerm(
        func=mdp.is_terminated_term,
        params={
            "term_keys": "peg_missed_hole", 
        },
        weight=TaskParams.peg_missed_hole_weight,
    )

    # penalize_peg_missed_hole = RewTerm(
    #     func=mdp.penalize_peg_missed_hole,
    #     params={
    #         "hole_cfg": SceneEntityCfg("hole"),
    #         "object_cfg": SceneEntityCfg("object"),
    #         "termination_height": TaskParams.termination_height,
    #         "xy_margin": TaskParams.xy_margin,
    #         "xy_threshold": TaskParams.xy_threshold,
    #     },
    #     weight=TaskParams.peg_missed_hole_weight,
    # )
    ######################################################

    # Action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=TaskParams.action_rate_weight)

    # Contact Wrench Penalty
    force_penalty = RewTerm(
        func=mdp.penalize_contact_forces,
        weight=TaskParams.force_penalty_weight,
    )

    torque_penalty = RewTerm(
        func=mdp.penalize_contact_torque,
        weight=TaskParams.torque_penalty_weight,
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={
            "minimum_height": TaskParams.object_dropping_min_height, 
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    is_peg_inserted = DoneTerm(
        func=mdp.is_peg_inserted,
        params={
            "hole_cfg": SceneEntityCfg("hole"),
            "object_cfg": SceneEntityCfg("object"),
            "object_height": TaskParams.object_height,
            "xy_threshold": TaskParams.is_peg_centered_xy_threshold,
            "z_variability": TaskParams.is_peg_centered_z_variability,
        }
    )

    peg_missed_hole = DoneTerm(
        func=mdp.peg_missed_hole,
        params={
            "hole_cfg": SceneEntityCfg("hole"),
            "object_cfg": SceneEntityCfg("object"),
            "termination_height": TaskParams.termination_height,
            "xy_margin": TaskParams.xy_margin,
            "xy_threshold": TaskParams.xy_threshold,
        }
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, 
    #     params={
    #         "term_name": "action_rate", 
    #         "weight": TaskParams.action_rate_curriculum_weight, 
    #         "num_steps": TaskParams.action_rate_curriculum_num_steps,
    #     },
    # )


##
# Environment configuration
##

@configclass
class UR5e_PegInsertEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""
    # Scene settings
    scene: UR5e_PegInsertSceneCfg = UR5e_PegInsertSceneCfg(num_envs=4, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = TaskParams.decimation
        self.episode_length_s = TaskParams.episode_length_s
        # simulation settings
        self.sim.dt = TaskParams.dt  # 100Hz
        self.sim.render_interval = TaskParams.render_interval #self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_collision_stack_size = 4096 * 4096 * 100 # Was added due to an PhysX error: collisionStackSize buffer overflow detected