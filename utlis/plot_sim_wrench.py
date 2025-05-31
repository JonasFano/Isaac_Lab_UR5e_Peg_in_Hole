import pandas as pd
import matplotlib.pyplot as plt

# filename = "impedance_ctrl_peg_insert_2048_envs_v12"
# filename = "impedance_ctrl_peg_insert_2048_envs_v13"
filename = "impedance_ctrl_peg_insert_2048_envs_v14"
# filename = "impedance_ctrl_peg_insert_2048_envs_v15"
# filename = "impedance_ctrl_peg_insert_2048_envs_v16"

data_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/data/" 

# Load CSV
df = pd.read_csv(data_path + filename + ".csv")

# start_timestep = 42
# end_timestep = 90
start_timestep = 62
end_timestep = 109
gravity_compensation = 14.5     # Newtons to add to z-force

# Filter by timestep range
df_filtered = df[(df['timestep'] >= start_timestep) & (df['timestep'] <= end_timestep)].copy()

# Apply gravity compensation to z-force
df_filtered['wrench_2'] += gravity_compensation

# Define labels for plotting
force_labels = {
    'wrench_0': 'Force fx',
    'wrench_1': 'Force fy',
    'wrench_2': 'Force fz'
}
torque_labels = {
    'wrench_3': 'Torque tx',
    'wrench_4': 'Torque ty',
    'wrench_5': 'Torque tz'
}

# Plot forces
# plt.figure()
# for col, label in force_labels.items():
#     plt.plot(df_filtered['timestep'], df_filtered[col], label=label)
# # plt.title(f'Forces from timestep {start_timestep} to {end_timestep}')
# plt.xlabel('Timestep')
# plt.ylabel('Force [N]')
# # plt.legend()
# plt.legend(loc='upper right')  # Other options: 'lower left', 'best', etc.

# plt.grid(True)
# plt.savefig("sim_force.pdf")

plt.rcParams.update({
    'font.size': 16,           # Controls default text size
    'axes.labelsize': 17,      # Axis label size
    'xtick.labelsize': 15,     # X tick label size
    'ytick.labelsize': 15,     # Y tick label size
    'legend.fontsize': 13,     # Legend font size
})

# # Plot torques
# Plot torques with tx highlighted
fig, ax1 = plt.subplots(figsize=(8, 6.1))

for col, label in torque_labels.items():
    if col == 'wrench_3':  # tx
        ax1.plot(df_filtered['timestep'], df_filtered[col], label=label, color='tab:blue', linewidth=1.2)
    if col == 'wrench_4':
        ax1.plot(df_filtered['timestep'], df_filtered[col], label=label, color='tab:orange', linestyle='--', linewidth=1)
    if col == 'wrench_5':
        ax1.plot(df_filtered['timestep'], df_filtered[col], label=label, color='tab:green', linestyle='--', linewidth=1)
plt.xlabel('Timestep')
plt.ylabel('Torque [Nm]')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("sim_torque.pdf")
plt.show()



fig, ax1 = plt.subplots(figsize=(8, 6.1))

# Plot forces with fz highlighted
for col, label in force_labels.items():
    if col == 'wrench_2':  # fz
        ax1.plot(df_filtered['timestep'], df_filtered[col], label=label, color='tab:green', linewidth=1.4)
    else:
        ax1.plot(df_filtered['timestep'], df_filtered[col], label=label, alpha=0.6, linestyle='--', linewidth=0.9)
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Force [N]')
ax1.grid(True)

# Plot z-position
ax2 = ax1.twinx()
ax2.plot(df_filtered['timestep'], df_filtered['tcp_position_2'], label='TCP z', color='tab:purple', linewidth=1.4)
ax2.set_ylabel('TCP z-position [m]', color='tab:purple')
ax2.tick_params(axis='y', labelcolor='tab:purple')

# Combine legends from both axes if desired
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

fig.tight_layout()
plt.savefig("sim_force.pdf")
plt.show()
