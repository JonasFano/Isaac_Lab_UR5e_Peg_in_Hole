import os
import numpy as np
import matplotlib.pyplot as plt

# OBS_LOG_PATH = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/data/obs_log_v2.npy"
OBS_LOG_PATH = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/data/obs_log_v5.npy"

def load_logged_obs(filepath):
    obs = []
    with open(filepath, 'rb') as f:
        while True:
            try:
                obs.append(np.load(f))
            except EOFError:
                break
    if not obs:
        raise RuntimeError("No observations loaded — check if file is empty or corrupted.")
    return np.stack(obs)  # Shape: (timesteps, num_envs, obs_dim)

def check_obs_stats(obs: np.ndarray):
    """
    Print per-environment observation statistics (min, max, NaN, inf).
    Args:
        obs: np.ndarray of shape (timesteps, num_envs, obs_dim)
    """
    timesteps, num_envs, obs_dim = obs.shape
    print(f"Loaded observations shape: {obs.shape}")

    for env_idx in range(num_envs):
        env_obs = obs[:, env_idx, :]  # (timesteps, obs_dim)
        min_val = np.min(env_obs)
        max_val = np.max(env_obs)
        has_nan = np.isnan(env_obs).any()
        has_inf = np.isinf(env_obs).any()

        if has_nan or has_inf or min_val < -1e3 or max_val > 1e3:
            print(f"[Env {env_idx}] min: {min_val:.3f}, max: {max_val:.3f}, NaN: {has_nan}, Inf: {has_inf}")

def identify_nan_categories(obs: np.ndarray):
    """
    Print which observation category (TCP pose, hole pose, previous action) contains NaN values, per env.
    """
    _, num_envs, obs_dim = obs.shape
    obs_labels = ["TCP pose"] * 7 + ["Hole pose"] * 7 + ["Previous action"] * 6

    for env_idx in range(num_envs):
        env_obs = obs[:, env_idx, :]
        for dim in range(obs_dim):
            if np.isnan(env_obs[:, dim]).any():
                print(f"[Env {env_idx}] NaN in obs dim {dim} ({obs_labels[dim]})")

def plot_obs_range_per_env(obs: np.ndarray):
    env_mins = np.min(obs, axis=(0, 2))
    env_maxs = np.max(obs, axis=(0, 2))

    plt.figure(figsize=(14, 6))
    plt.plot(env_mins, label="Min obs per env", color='blue')
    plt.plot(env_maxs, label="Max obs per env", color='red')
    plt.title("Observation range per environment")
    plt.xlabel("Environment index")
    plt.ylabel("Observation value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_env_obs(obs: np.ndarray, env_idx: int = 0):
    env_obs = obs[:, env_idx, :]
    obs_dim = env_obs.shape[1]

    plt.figure(figsize=(12, 6))
    for i in range(obs_dim):
        plt.plot(env_obs[:, i], label=f'Obs dim {i}')
    plt.title(f'Observations over time for env {env_idx}')
    plt.xlabel('Timestep')
    plt.ylabel('Observation value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    obs = load_logged_obs(OBS_LOG_PATH)
    print(obs[22026, 50, :])
    print(obs[22027, 50, :])
    check_obs_stats(obs)
    identify_nan_categories(obs)
    plot_obs_range_per_env(obs)
    # plot_env_obs(obs, env_idx=50)
