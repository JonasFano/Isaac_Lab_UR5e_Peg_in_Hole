from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from . import mdp
import os
import math

from taskparameters import TaskParams

##
# Scene definition
##

MODEL_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), "scene_models")

@configclass
class UR5e_PegInHoleSceneCfg(InteractiveSceneCfg):
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
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),  
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.175, -0.175, 0.0), 
            joint_pos={
                "shoulder_pan_joint": TaskParams.robot_initial_joint_pos[0], 
                "shoulder_lift_joint": TaskParams.robot_initial_joint_pos[1], 
                "elbow_joint": TaskParams.robot_initial_joint_pos[2], 
                "wrist_1_joint": TaskParams.robot_initial_joint_pos[3], 
                "wrist_2_joint": TaskParams.robot_initial_joint_pos[4], 
                "wrist_3_joint": TaskParams.robot_initial_joint_pos[5], 
                "joint_left": TaskParams.robot_initial_joint_pos[6], # or robotiq_hande_left_finger_joint
                "joint_right": TaskParams.robot_initial_joint_pos[7], # or robotiq_hande_right_finger_joint
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
                ############### Stiffness 10000000 ###############
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
                    "shoulder_pan_joint": TaskParams.shoulder_pan_damping,
                    "shoulder_lift_joint": TaskParams.shoulder_lift_damping,
                    "elbow_joint": TaskParams.elbow_damping,
                    "wrist_1_joint": TaskParams.wrist_1_damping,
                    "wrist_2_joint": TaskParams.wrist_2_damping,
                    "wrist_3_joint": TaskParams.wrist_3_damping,
                    "joint_left": TaskParams.gripper_damping,
                    "joint_right": TaskParams.gripper_damping,
                }
            )
        }
    )

    # Add peg and hole objects on the table
    object = RigidObjectCfg(
        prim_path = "{ENV_REGEX_NS}/object", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(MODEL_PATH, "cranfield_model/Cranfield parts - BolzenKleinEckig.usd"), 
            scale=TaskParams.object_scale,
            rigid_props=RigidBodyPropertiesCfg(
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
            mass_props=MassPropertiesCfg(
                mass=TaskParams.object_init_mass,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ), 
        init_state=RigidObjectCfg.InitialStateCfg(pos=TaskParams.object_init_pos, lin_vel=(0.0, 0.0, 0.0)),
    )

    hole = RigidObjectCfg(
        prim_path = "{ENV_REGEX_NS}/hole", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(MODEL_PATH, "cranfield_model/Cranfield parts - CranfieldBase.usd"),
            mass_props=MassPropertiesCfg(
                mass=TaskParams.hole_init_mass,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ), 
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TaskParams.hole_init_pos,
        ),
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.7071, 0.0, 0.0, 0.7071)),
    )



##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="wrist_3_link", 
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.25, 0.35), 
    #         pos_y=(0.3, 0.4), 
    #         pos_z=(0.25, 0.35), 
    #         roll=(0.0, 0.0),
    #         pitch=(math.pi, math.pi),  # depends on end-effector axis
    #         yaw=(-3.14, 3.14), # (0.0, 0.0), # y
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Set actions
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_left", "joint_right"],
        open_command_expr={"joint_left": TaskParams.gripper_open[0], "joint_right": TaskParams.gripper_open[1]},
        close_command_expr={"joint_left": TaskParams.gripper_joint_pos_close[0], "joint_right": TaskParams.gripper_joint_pos_close[1]},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_left", "joint_right"]),},
        )

        tcp_pose = ObsTerm(
            func=mdp.get_current_tcp_pose,
            params={"gripper_offset": TaskParams.gripper_offset, "robot_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
        )

        ee_wrench_b = ObsTerm(
            func=mdp.body_incoming_wrench_transform,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])}
        ) # Small force/torque if not in contact, small but noticeable changes when moving, gets big when in contact

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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    hole_ee_distance = RewTerm(
        func=mdp.hole_ee_distance, 
        params={"std": TaskParams.hole_ee_distance_std}, 
        weight=TaskParams.hole_ee_distance_weight,
    )

    orientation_tracking = RewTerm(
        func=mdp.object_hole_orientation_error, 
        params={
            "hole_cfg": SceneEntityCfg("hole"),
            "object_cfg": SceneEntityCfg("object"),
        }, 
        weight=TaskParams.orientation_tracking_weight
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=TaskParams.action_rate_weight)



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


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={
            "term_name": "action_rate", 
            "weight": TaskParams.action_rate_curriculum_weight, 
            "num_steps": TaskParams.action_rate_curriculum_num_steps,
        },
    )


##
# Environment configuration
##

@configclass
class UR5e_PegInHoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""
    # Scene settings
    scene: UR5e_PegInHoleSceneCfg = UR5e_PegInHoleSceneCfg(num_envs=4, env_spacing=2.5)

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
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_collision_stack_size = 4096 * 4096 * 120 # Was added due to an PhysX error: collisionStackSize buffer overflow detected