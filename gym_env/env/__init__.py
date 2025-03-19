import gymnasium as gym
from . import agents, ik_rel_env_cfg, ik_rel_env_cfg_domain_rand

# Register Gym environments.

gym.register(
    id="Peg-in-hole-IK",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.RelIK_UR5e_PegInHoleEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)


gym.register(
    id="Peg-in-hole-IK-Domain-Rand",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg_domain_rand.RelIK_UR5e_Domain_Rand_PegInHoleEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

