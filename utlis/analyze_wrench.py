import pandas as pd
import numpy as np

# filename = "impedance_ctrl_peg_insert_2048_envs_v12"
# filename = "impedance_ctrl_peg_insert_2048_envs_v13"
# filename = "impedance_ctrl_peg_insert_2048_envs_v14"
# filename = "impedance_ctrl_peg_insert_2048_envs_v15"
# filename = "impedance_ctrl_peg_insert_2048_envs_v16"
# filename = "impedance_ctrl_peg_insert_2048_envs_v17"
filename = "impedance_ctrl_peg_insert_2048_envs_v18"

data_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/data/" 

# Load CSV
df = pd.read_csv(data_path + filename + ".csv")

# Helper: Check if a row is an episode separator (all zeros in wrench or action)
def is_episode_separator(row):
    return (row[['wrench_0', 'wrench_1', 'wrench_2', 'wrench_3', 'wrench_4', 'wrench_5']].abs().sum() == 0 or
            row[['actions_0', 'actions_1', 'actions_2']].abs().sum() == 0)

# Find episode boundaries
episode_indices = df[df.apply(is_episode_separator, axis=1)].index.tolist()
episode_indices.append(len(df))  # add end of last episode

print(len(episode_indices))

# Compute metrics per episode
episode_metrics = []
for i in range(50):
    start = episode_indices[i] + 1  # skip the separator row
    end = episode_indices[i + 1]
    ep_df = df.iloc[start:end]
    if ep_df.empty:
        continue

    forces = ep_df[['wrench_0', 'wrench_1', 'wrench_2']].to_numpy()
    torques = ep_df[['wrench_3', 'wrench_4', 'wrench_5']].to_numpy()
    
    force_magnitudes = np.linalg.norm(forces, axis=1)
    torque_magnitudes = np.linalg.norm(torques, axis=1)

    metrics = {
        'mean_force': force_magnitudes.mean(),
        'max_force': np.abs(forces).max(),
        'mean_z_force': ep_df['wrench_2'].mean(),
        'mean_torque': torque_magnitudes.mean(),
        'max_torque': np.abs(torques).max(),  # max absolute torque
    }
    episode_metrics.append(metrics)

# Aggregate across episodes
all_metrics = pd.DataFrame(episode_metrics)
summary = {
    'mean_force_across_episodes': all_metrics['mean_force'].mean(),
    'mean_max_force_across_episodes': all_metrics['max_force'].mean(),
    'mean_z_force_across_episodes': all_metrics['mean_z_force'].mean(),
    'mean_torque_across_episodes': all_metrics['mean_torque'].mean(),
    'mean_max_torque_across_episodes': all_metrics['max_torque'].mean(),
}

# Print summary
for k, v in summary.items():
    print(f'{k}: {v:.4f}')
