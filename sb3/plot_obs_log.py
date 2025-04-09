import os
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.subplots import make_subplots

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


def plot_tcp_pose_for_env(obs: np.ndarray, env_idx: int = 50, start: int = 0, end: int = None):
    """
    Plot TCP pose (position + quaternion) over time for a specific environment.
    
    Args:
        obs: np.ndarray of shape (timesteps, num_envs, obs_dim)
        env_idx: which environment to plot
        start: starting timestep
        end: ending timestep (exclusive)
    """
    max_timesteps = obs.shape[0]
    if end is None or end > max_timesteps:
        end = max_timesteps
    if start >= end:
        raise ValueError(f"Start timestep ({start}) must be less than end timestep ({end}).")

    tcp_pose = obs[start:end, env_idx, 0:7]  # [timesteps, 7]
    labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

    plt.figure(figsize=(12, 6))
    for i in range(7):
        plt.plot(np.arange(start, end), tcp_pose[:, i], label=labels[i])
    plt.title(f'TCP Pose over time for env {env_idx}')
    plt.xlabel('Timestep')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_tcp_xyz_for_env(obs: np.ndarray, env_idx: int = 50, start: int = 0, end: int = None):
    """
    Plot TCP position (x, y, z) over time for a specific environment in separate subplots.
    
    Args:
        obs: np.ndarray of shape (timesteps, num_envs, obs_dim)
        env_idx: which environment to plot
        start: starting timestep
        end: ending timestep (exclusive)
    """
    max_timesteps = obs.shape[0]
    if end is None or end > max_timesteps:
        end = max_timesteps
    if start >= end:
        raise ValueError(f"Start timestep ({start}) must be less than end timestep ({end}).")

    tcp_pose = obs[start:end, env_idx, 0:3]  # Only x, y, z
    labels = ['x', 'y', 'z']

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for i in range(3):
        axes[i].plot(np.arange(start, end), tcp_pose[:, i], label=labels[i])
        axes[i].set_ylabel(f'{labels[i]} position')
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel('Timestep')
    fig.suptitle(f'TCP Position (x, y, z) over time for env {env_idx}', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_tcp_xyz_for_env_plotly(obs: np.ndarray, env_idx: int = 50, start: int = 0, end: int = None):
    """
    Plot TCP position (x, y, z) over time for a specific environment using Plotly in separate subplots.

    Args:
        obs: np.ndarray of shape (timesteps, num_envs, obs_dim)
        env_idx: which environment to plot
        start: starting timestep
        end: ending timestep (exclusive)
    """
    max_timesteps = obs.shape[0]
    if end is None or end > max_timesteps:
        end = max_timesteps
    if start >= end:
        raise ValueError(f"Start timestep ({start}) must be less than end timestep ({end}).")

    tcp_pose = obs[start:end, env_idx, 0:3]  # Only x, y, z
    timesteps = np.arange(start, end)
    labels = ['x', 'y', 'z']

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=[f'{label} position' for label in labels])

    for i in range(3):
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=tcp_pose[:, i],
            mode='lines',
            name=labels[i],
        ), row=i+1, col=1)

        fig.update_yaxes(title_text=f"{labels[i]} (m)", row=i+1, col=1)

    fig.update_xaxes(title_text="Timestep", row=3, col=1)
    fig.update_layout(height=1400, width=3000,
                      xaxis_title='Timestep',
                      yaxis_title='Position (m)',
                      showlegend=False,
                      hovermode="x unified",)
    fig.show()


if __name__ == "__main__":
    obs = load_logged_obs(OBS_LOG_PATH)
    print(obs[22026, 50, :])
    print(obs[22027, 50, :])
    check_obs_stats(obs)
    identify_nan_categories(obs)
    # plot_obs_range_per_env(obs)
    # plot_env_obs(obs, env_idx=50)
    # plot_tcp_pose_for_env(obs, env_idx=50, start=20000, end=22030)
    plot_tcp_xyz_for_env_plotly(obs, env_idx=50, start=15000, end=obs.shape[0])


