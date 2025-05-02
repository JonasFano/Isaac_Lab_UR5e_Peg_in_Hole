import gymnasium as gym
from . import agents, ik_rel_peg_insert_env_cfg, ik_rel_peg_insert_env_cfg_franka
from . import imp_ctrl_peg_insert_env_cfg, imp_ctrl_peg_insert_env_cfg_franka

# Register Gym environments.


gym.register(
    id="UR5e-Peg-Insert-IK",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_rel_peg_insert_env_cfg.RelIK_UR5e_PegInsertEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)


gym.register(
    id="Franka-Peg-Insert-IK",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_rel_peg_insert_env_cfg_franka.RelIK_Franka_PegInsertEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)



gym.register(
    id="UR5e-Peg-Insert-Impedance-Ctrl",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": imp_ctrl_peg_insert_env_cfg.ImpCtrl_UR5e_PegInsertEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)


gym.register(
    id="Franka-Peg-Insert-Impedance-Ctrl",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": imp_ctrl_peg_insert_env_cfg_franka.ImpCtrl_Franka_PegInsertEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)