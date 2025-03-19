from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from . import peg_in_hole_env_cfg_domain_rand

from taskparameters import TaskParams

@configclass
class RelIK_UR5e_Domain_Rand_PegInHoleEnvCfg(peg_in_hole_env_cfg_domain_rand.UR5e_Domain_Rand_PegInHoleEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=TaskParams.gripper_offset,
                    ),
                ),
            ],
        )

        # print(self.commands.object_pose) # Do not show current end-effector frame
        # self.commands.object_pose.current_pose_visualizer_cfg.markers['frame'].visible = False

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(command_type=TaskParams.command_type, use_relative_mode=TaskParams.use_relative_mode, ik_method=TaskParams.ik_method),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TaskParams.gripper_offset),
            scale=TaskParams.action_scale,
            debug_vis=False  # Enable debug visualization, set to False for production
        )


@configclass
class RelIK_UR5e_Domain_Rand_PegInHoleEnvCfg_PLAY(RelIK_UR5e_Domain_Rand_PegInHoleEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

