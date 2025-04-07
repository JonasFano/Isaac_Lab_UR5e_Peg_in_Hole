import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your binary wrench min/max log file
# log_path = Path(__file__).resolve().parents[0] / "../data/wrench_log_minmax.bin" # Without force/torque clamping
log_path = Path(__file__).resolve().parents[0] / "../data/wrench_log_minmax_v1.bin" # Without force/torque clamping
# log_path = Path(__file__).resolve().parents[0] / "../data/wrench_log_minmax_v2.bin" # With force/torque clamping to +/- 10000

# Load the binary file as float32 and reshape
data = np.fromfile(log_path, dtype=np.float32)

# Ensure it has 12 columns (6 min + 6 max)
assert data.size % 12 == 0, "Data size is not divisible by 12!"
wrench_data = data.reshape(-1, 12)

# Split into min and max force/torque
min_forces = wrench_data[:, 0:3]   # Fx, Fy, Fz (min)
min_torques = wrench_data[:, 3:6]  # Tx, Ty, Tz (min)
max_forces = wrench_data[:, 6:9]   # Fx, Fy, Fz (max)
max_torques = wrench_data[:, 9:12] # Tx, Ty, Tz (max)

do_not_plot = 1000

# Time axis (just indices)
timesteps = np.arange(len(wrench_data) - do_not_plot)

min_forces = min_forces[:-do_not_plot]
max_forces = max_forces[:-do_not_plot]
min_torques = min_torques[:-do_not_plot]
max_torques = max_torques[:-do_not_plot]

# Plot
plt.figure(figsize=(14, 10))

# Min Forces
plt.subplot(4, 1, 1)
plt.plot(timesteps, min_forces[:, 0], label="Fx min")
plt.plot(timesteps, min_forces[:, 1], label="Fy min")
plt.plot(timesteps, min_forces[:, 2], label="Fz min")
plt.title("Min End-Effector Forces")
plt.ylabel("Force (N)")
plt.legend()
plt.grid(True)

# Max Forces
plt.subplot(4, 1, 2)
plt.plot(timesteps, max_forces[:, 0], label="Fx max")
plt.plot(timesteps, max_forces[:, 1], label="Fy max")
plt.plot(timesteps, max_forces[:, 2], label="Fz max")
plt.title("Max End-Effector Forces")
plt.ylabel("Force (N)")
plt.legend()
plt.grid(True)

# Min Torques
plt.subplot(4, 1, 3)
plt.plot(timesteps, min_torques[:, 0], label="Tx min")
plt.plot(timesteps, min_torques[:, 1], label="Ty min")
plt.plot(timesteps, min_torques[:, 2], label="Tz min")
plt.title("Min End-Effector Torques")
plt.ylabel("Torque (Nm)")
plt.legend()
plt.grid(True)

# Max Torques
plt.subplot(4, 1, 4)
plt.plot(timesteps, max_torques[:, 0], label="Tx max")
plt.plot(timesteps, max_torques[:, 1], label="Ty max")
plt.plot(timesteps, max_torques[:, 2], label="Tz max")
plt.title("Max End-Effector Torques")
plt.xlabel("Timestep")
plt.ylabel("Torque (Nm)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
