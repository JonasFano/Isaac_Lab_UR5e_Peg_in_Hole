from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from gym_env.env.controller.impedance_control_cfg import ImpedanceControllerCfg
from gym_env.env.mdp.actions.actions_cfg import ImpedanceControllerActionCfg

from . import peg_insert_env_cfg_impedance

from taskparameters_peginsert_impedance import TaskParams

@configclass
class ImpCtrl_UR5e_PegInsertEnvCfg(peg_insert_env_cfg_impedance.UR5e_PegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Visualize TCP/End-effector frame
        base_marker_cfg = FRAME_MARKER_CFG.copy()
        base_marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.25)
        base_marker_cfg.prim_path = "/Visuals/BaseFrame"

        self.scene.world_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/base_link",
            debug_vis=True,
            visualizer_cfg=base_marker_cfg,
            target_frames=[  # Add at least one target frame
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot/base_link",
                    name="base_frame",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                )
            ],
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/base_link",
            debug_vis=False, # True to visualize ee frame or False
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

        self.actions.arm_action = ImpedanceControllerActionCfg(
            asset_name="robot",
            joint_names=TaskParams.joint_names,
            body_name=TaskParams.ee_body_name,
            controller_cfg=ImpedanceControllerCfg(
                gravity_compensation=TaskParams.gravity_compensation,
                coriolis_centrifugal_compensation=TaskParams.coriolis_centrifugal_compensation,
                inertial_dynamics_decoupling=TaskParams.inertial_dynamics_decoupling,
                max_torque_clamping=TaskParams.max_torque_clamping,
                stiffness=TaskParams.stiffness,
                damping=TaskParams.damping,
                damping_ratio=TaskParams.damping_ratio,
                damping_ratio_z=TaskParams.damping_ratio_z,
            ),
            action_scale=TaskParams.action_scale,
            tcp_offset=TaskParams.gripper_offset,
        )


@configclass
class ImpCtrl_UR5e_PegInsertEnvCfg_PLAY(ImpCtrl_UR5e_PegInsertEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

