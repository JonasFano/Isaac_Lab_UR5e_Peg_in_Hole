import pandas as pd
import matplotlib.pyplot as plt

# Load the data

filename = "tcp_wrench_log_v4"
# filename = "tcp_wrench_log_v5"

path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Peg_in_Hole/real_world_execution/"

df = pd.read_csv(path + filename + '.csv')

# Extract force and torque columns
force_cols = ['fx', 'fy', 'fz']
torque_cols = ['tx', 'ty', 'tz']

# Plot force
plt.figure()
for col in force_cols:
    plt.plot(df['timestamp'], df[col], label=col)
plt.title('Force over Time')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.grid(True)

# Plot torque
plt.figure()
for col in torque_cols:
    plt.plot(df['timestamp'], df[col], label=col)
plt.title('Torque over Time')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.legend()
plt.grid(True)

# Print max absolute values
max_force = df[force_cols].abs().max()
max_torque = df[torque_cols].abs().max()

print("Max absolute force components:")
print(max_force)
print("\nMax absolute torque components:")
print(max_torque)

plt.show()
