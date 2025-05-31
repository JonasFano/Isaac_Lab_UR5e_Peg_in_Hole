import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('learning_curve_impedance_control.csv')

# Convert Step to numeric (in case it's read as string)
df['Step'] = pd.to_numeric(df['Step'])

# Plot setup
plt.figure(figsize=(10, 6))

plt.rcParams.update({
    'font.size': 16,           # Controls default text size
    'axes.labelsize': 20,      # Axis label size
    'xtick.labelsize': 16,     # X tick label size
    'ytick.labelsize': 16,     # Y tick label size
    'legend.fontsize': 18,     # Legend font size
})


# Plot VeryHighPen-DR3
x = df['Step']
y = df['VeryHighPen-DR3 - rollout/ep_rew_mean']
ymin = df['VeryHighPen-DR3 - rollout/ep_rew_mean__MIN']
ymax = df['VeryHighPen-DR3 - rollout/ep_rew_mean__MAX']
plt.plot(x, y, label='VeryHighPen-DR3', color='blue')
plt.fill_between(x, ymin, ymax, color='blue', alpha=0.2)

# Plot MedPen-HoleNoise if available (non-empty)
if not df['MedPen-HoleNoise - rollout/ep_rew_mean'].isnull().all():
    y2 = df['MedPen-HoleNoise - rollout/ep_rew_mean']
    ymin2 = df['MedPen-HoleNoise - rollout/ep_rew_mean__MIN']
    ymax2 = df['MedPen-HoleNoise - rollout/ep_rew_mean__MAX']
    plt.plot(x, y2, label='MedPen-HoleNoise', color='green', linestyle='-')
    plt.fill_between(x, ymin2, ymax2, color='green', alpha=0.2)

# Labels and legend
plt.xlabel('Training Steps')
plt.ylabel('Mean Episode Reward')
# plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve_impedance_control.pdf")
plt.show()
