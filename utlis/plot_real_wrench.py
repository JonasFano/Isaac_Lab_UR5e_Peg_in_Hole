import pandas as pd
import matplotlib.pyplot as plt

# Load the data
filename = "tcp_wrench_log_success" # Successful insertion

path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/real_world_execution/"

df = pd.read_csv(path + filename + '.csv')

# Extract force and torque columns
force_cols = ['fx', 'fy', 'fz']
torque_cols = ['tx', 'ty', 'tz']
time_col = 'timestamp'
position_col = 'tcp_z'

plt.rcParams.update({
    'font.size': 16,           # Controls default text size
    'axes.labelsize': 16,      # Axis label size
    'xtick.labelsize': 12,     # X tick label size
    'ytick.labelsize': 12,     # Y tick label size
    'legend.fontsize': 14,     # Legend font size
})

# Plot force
# plt.figure()
# for col in force_cols:
#     plt.plot(df['timestamp'], df[col], label=col)
# # plt.title('Force over Time')
# plt.xlabel('Time [s]')
# plt.ylabel('Force [N]')
# plt.legend()
# plt.grid(True)
# plt.savefig("real_force.pdf")

# # Plot torque
# plt.figure()
# for col in torque_cols:
#     plt.plot(df['timestamp'], df[col], label=col)
# # plt.title('Torque over Time')
# plt.xlabel('Time [s]')
# plt.ylabel('Torque [Nm]')
# plt.legend()
# plt.grid(True)

# # Print max absolute values
# max_force = df[force_cols].abs().max()
# max_torque = df[torque_cols].abs().max()

# print("Max absolute force components:")
# print(max_force)
# print("\nMax absolute torque components:")
# print(max_torque)
# plt.savefig("real_torque.pdf")

# # Plot TCP z-position
# plt.figure()
# plt.plot(df[time_col], df[position_col], label='tcp_z', color='purple')
# plt.xlabel('Time [s]')
# plt.ylabel('TCP z-position [m]')
# plt.title('TCP z-position over time')
# plt.grid(True)
# plt.legend()
# plt.savefig("real_position.pdf")

# # Print max values
# max_force = df[force_cols].abs().max()
# max_torque = df[torque_cols].abs().max()
# print("Max absolute force components:")
# print(max_force)
# print("\nMax absolute torque components:")
# print(max_torque)


# # Plot only z-force (fz)
# plt.figure()
# plt.plot(df[time_col], df['fz'], label='fz', color='tab:green')
# plt.xlabel('Time [s]')
# plt.ylabel('Force z [N]')
# plt.legend()
# plt.grid(True)
# plt.savefig("real_force_z.pdf")


plt.figure()

# Highlight tx (blue, solid), fade ty and tz (orange and green, dashed)
plt.plot(df[time_col], df['tx'], label='tx', color='tab:blue', linewidth=1.0)
# plt.plot(df[time_col], df['ty'], label='ty', color='tab:orange', alpha = 0.8, linestyle='--', linewidth=1)
# plt.plot(df[time_col], df['tz'], label='tz', color='tab:green', alpha = 0.8, linestyle='--', linewidth=1)

plt.xlabel('Time [s]')
plt.ylabel('Torque about x-axis [Nm]')
# plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig("real_torque.pdf")
plt.show()



fig, ax1 = plt.subplots()

# Plot fz on left y-axis
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Force in z-direction [N]', color='tab:green')
ax1.plot(df['timestamp'], df['fz'], label='fz', color='tab:green')
ax1.tick_params(axis='y', labelcolor='tab:green')
ax1.grid(True)

# Plot tcp_z on right y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('TCP z-position [m]', color='tab:purple')
ax2.plot(df['timestamp'], df['tcp_z'], label='tcp_z', color='tab:purple')
ax2.tick_params(axis='y', labelcolor='tab:purple')

# plt.title('Force z and TCP z-position over time')
fig.tight_layout()
plt.savefig("real_force.pdf")


plt.show()

