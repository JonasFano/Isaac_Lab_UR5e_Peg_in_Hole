from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg
from . import mdp
import os


##
# Scene definition
##

MODEL_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), "scene_models")

@configclass
class UR5e_Domain_Rand_PegInHoleSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object."""
    # articulation
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(MODEL_PATH, "ur5e_robotiq_new.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            activate_contact_sensors=True,), 
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.175, -0.175, 0.0), 
            joint_pos={
                "shoulder_pan_joint": 1.3, 
                "shoulder_lift_joint": -2.0, 
                "elbow_joint": 2.0, 
                "wrist_1_joint": -1.5, 
                "wrist_2_joint": -1.5, 
                "wrist_3_joint": 3.14, 
                "joint_left": 0.0, 
                "joint_right": 0.0,
            }
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # Match all joints
                velocity_limit={
                    "shoulder_pan_joint": 180.0,
                    "shoulder_lift_joint": 180.0,
                    "elbow_joint": 180.0,
                    "wrist_1_joint": 180.0,
                    "wrist_2_joint": 180.0,
                    "wrist_3_joint": 180.0,
                    "joint_left": 1000000.0,
                    "joint_right": 1000000.0,
                },
                effort_limit={
                    "shoulder_pan_joint": 87.0,
                    "shoulder_lift_joint": 87.0,
                    "elbow_joint": 87.0,
                    "wrist_1_joint": 87.0,
                    "wrist_2_joint": 87.0,
                    "wrist_3_joint": 87.0,
                    "joint_left": 200.0,
                    "joint_right": 200.0,
                },
                # stiffness={
                #     "shoulder_pan_joint": 261.79941,
                #     "shoulder_lift_joint": 261.79941,
                #     "elbow_joint": 261.79941,
                #     "wrist_1_joint": 261.79941,
                #     "wrist_2_joint": 261.79941,
                #     "wrist_3_joint": 261.79941,
                #     "joint_left": 3000.0,
                #     "joint_right": 3000.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 26.17994,
                #     "shoulder_lift_joint": 26.17994,
                #     "elbow_joint": 26.17994,
                #     "wrist_1_joint": 26.17994,
                #     "wrist_2_joint": 26.17994,
                #     "wrist_3_joint": 26.17994,
                #     "joint_left": 800.0,
                #     "joint_right": 800.0,
                # }
                stiffness={
                    "shoulder_pan_joint": 1000.0,
                    "shoulder_lift_joint": 1000.0,
                    "elbow_joint": 1000.0,
                    "wrist_1_joint": 1000.0,
                    "wrist_2_joint": 1000.0,
                    "wrist_3_joint": 1000.0,
                    "joint_left": 3000.0,
                    "joint_right": 3000.0,
                },
                damping={
                    "shoulder_pan_joint": 121.66,
                    "shoulder_lift_joint": 183.23,
                    "elbow_joint": 96.54,
                    "wrist_1_joint": 69.83,
                    "wrist_2_joint": 69.83,
                    "wrist_3_joint": 27.42,
                    "joint_left": 500.0,
                    "joint_right": 500.0,
                }
            )
        }
    )

    # Add peg and hole objects on the table
    object = RigidObjectCfg(
        prim_path = "{ENV_REGEX_NS}/Object", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(MODEL_PATH, "cranfield_model/Cranfield parts - BolzenKleinEckig.usd"), 
            scale=(0.92, 0.92, 1),
            mass_props=MassPropertiesCfg(
                mass=1.0,
            ),
        ), 
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.04, 0.35, 0.0), lin_vel=(0.0, 0.0, 0.0)),
    )

    hole = RigidObjectCfg(
        prim_path = "{ENV_REGEX_NS}/hole", 
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(MODEL_PATH, "cranfield_model/Cranfield parts - CranfieldBase.usd")), 
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.3, 0.0)),
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
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="wrist_3_link", 
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.25, 0.35), pos_y=(0.3, 0.4), pos_z=(0.25, 0.35), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Set actions
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_left", "joint_right"],
        open_command_expr={"joint_left": 0.0, "joint_right": 0.0},
        close_command_expr={"joint_left": 0.02, "joint_right": 0.02},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomize the object position 
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="object"),
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, 
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            "mass_distribution_params": (0.1, 1.5),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        }
    )

    randomize_robot_gains = EventTerm(
        func=mdp.randomize_actuator_gains_custom,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.9, 1.1),
            "operation_stiffness": "scale",
            "operation_damping": "scale",
            "distribution_stiffness": "uniform",
            "distribution_damping": "uniform",
        }
    )

    randomize_gripper_gains = EventTerm(
        func=mdp.randomize_actuator_gains_custom,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_left", "joint_right"]),
            "stiffness_distribution_params": (2500, 3500),
            "damping_distribution_params": (300, 700),
            "operation_stiffness": "scale",
            "operation_damping": "abs",
            "distribution_stiffness": "uniform",
            "distribution_damping": "uniform",
        }
    )

    randomize_gripper_fingers_friction_coefficients = EventTerm(
        func=mdp.randomize_friction_coefficients,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["finger_left", "finger_right"]),
            "static_friction_distribution_params": (0.5, 1.2), #(0.1, 1.5),
            "dynamic_friction_distribution_params": (0.4, 1.0), #(0.05, 1.2),
            "restitution_distribution_params": (0.2, 0.6), #(0.0, 1.0),
            "operation": "abs",
            "distribution": "uniform",
            "make_consistent": True,  # Ensure dynamic friction <= static friction
        }
    )

    randomize_object_friction_coefficients = EventTerm(
        func=mdp.randomize_friction_coefficients,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            "static_friction_distribution_params": (0.4, 0.8),
            "dynamic_friction_distribution_params": (0.3, 0.6),
            "restitution_distribution_params": (0.3, 0.7),
            "operation": "abs",
            "distribution": "uniform",
            "make_consistent": True,  # Ensure dynamic friction <= static friction
        }
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.09}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.09, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.09, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##

@configclass
class UR5e_Domain_Rand_PegInHoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""
    # Scene settings
    scene: UR5e_Domain_Rand_PegInHoleSceneCfg = UR5e_Domain_Rand_PegInHoleSceneCfg(num_envs=4, env_spacing=2.5)

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
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_collision_stack_size = 4096 * 4096 * 64 # Was added due to an PhysX error: collisionStackSize buffer overflow detected