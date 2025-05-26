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

start_timestep = 42
end_timestep = 90
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
plt.figure()
for col, label in force_labels.items():
    plt.plot(df_filtered['timestep'], df_filtered[col], label=label)
plt.title(f'Forces from timestep {start_timestep} to {end_timestep}')
plt.xlabel('Timestep')
plt.ylabel('Force [N]')
plt.legend()
plt.grid(True)

# Plot torques
plt.figure()
for col, label in torque_labels.items():
    plt.plot(df_filtered['timestep'], df_filtered[col], label=label)
plt.title(f'Torques from timestep {start_timestep} to {end_timestep}')
plt.xlabel('Timestep')
plt.ylabel('Torque [Nm]')
plt.legend()
plt.grid(True)

plt.show()
