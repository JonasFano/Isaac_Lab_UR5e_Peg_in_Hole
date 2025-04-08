import os
import numpy as np
import matplotlib.pyplot as plt

# LOG_PATH = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/data/action_log_v2.npy"
LOG_PATH = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/data/action_log_v5.npy"

def load_logged_actions(filepath):
    actions = []
    with open(filepath, 'rb') as f:
        while True:
            try:
                arr = np.load(f)
                if arr.ndim != 2 or arr.shape[1] != 6:
                    print("Warning: Unexpected shape in array, skipping.")
                    continue
                actions.append(arr)
            except EOFError:
                break
    if not actions:
        raise RuntimeError("No actions loaded — check if file is empty or corrupted.")
    return np.stack(actions)  # Shape: (timesteps, num_envs, action_dim)


def check_action_stats(actions: np.ndarray):
    """
    Print per-environment action statistics (min, max, NaN, inf).
    Args:
        actions: np.ndarray of shape (timesteps, num_envs, action_dim)
    """
    timesteps, num_envs, action_dim = actions.shape
    print(f"Loaded actions shape: {actions.shape}")

    count = 0

    for env_idx in range(num_envs):
        env_actions = actions[:, env_idx, :]  # (timesteps, action_dim)

        min_val = np.min(env_actions)
        max_val = np.max(env_actions)
        has_nan = np.isnan(env_actions).any()
        has_inf = np.isinf(env_actions).any()

        if has_nan or has_inf or min_val < -1e3 or max_val > 1e3:
            print(f"[Env {env_idx}] min: {min_val:.3f}, max: {max_val:.3f}, NaN: {has_nan}, Inf: {has_inf}")
            count =+ 1

    print("Invalid Actions: ", count)

def plot_action_range_per_env(actions: np.ndarray):
    """
    Plot the min and max action values for each environment.
    Args:
        actions: np.ndarray of shape (timesteps, num_envs, action_dim)
    """
    timesteps, num_envs, action_dim = actions.shape
    env_mins = np.min(actions, axis=(0, 2))  # (num_envs,)
    env_maxs = np.max(actions, axis=(0, 2))  # (num_envs,)

    plt.figure(figsize=(14, 6))
    plt.plot(env_mins, label="Min action per env", color='blue')
    plt.plot(env_maxs, label="Max action per env", color='red')
    plt.title("Action range per environment")
    plt.xlabel("Environment index")
    plt.ylabel("Action value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_env_actions(actions: np.ndarray, env_idx: int = 0):
    """
    Plot each action dimension over time for a given environment index.
    Args:
        actions: np.ndarray of shape (timesteps, num_envs, action_dim)
        env_idx: Index of the environment to plot (int)
    """
    timesteps = actions.shape[0]
    action_dim = actions.shape[2]
    env_actions = actions[:, env_idx, :]  # (timesteps, action_dim)

    plt.figure(figsize=(12, 6))
    for i in range(action_dim):
        plt.plot(env_actions[:, i], label=f'Action dim {i}')
    plt.title(f'Actions over time for env {env_idx}')
    plt.xlabel('Timestep')
    plt.ylabel('Action value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    actions = load_logged_actions(LOG_PATH)
    print(actions)
    check_action_stats(actions)
    plot_action_range_per_env(actions)
    # plot_env_actions(actions, env_idx=0)  # Optional: plot details for one env
