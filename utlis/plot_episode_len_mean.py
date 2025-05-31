import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('episode_len_mean_impedance_control.csv')  # Replace with your actual file name


# Convert Step to numeric
df['Step'] = pd.to_numeric(df['Step'])

# Create the plot
plt.figure(figsize=(10, 6))

plt.rcParams.update({
    'font.size': 18,           # Controls default text size
    'axes.labelsize': 20,      # Axis label size
    'xtick.labelsize': 16,     # X tick label size
    'ytick.labelsize': 16,     # Y tick label size
    'legend.fontsize': 18,     # Legend font size
})

# Plot VeryHighPen-DR3 (solid line)
x = df['Step']
y = df['VeryHighPen-DR3 - rollout/ep_len_mean']
ymin = df['VeryHighPen-DR3 - rollout/ep_len_mean__MIN']
ymax = df['VeryHighPen-DR3 - rollout/ep_len_mean__MAX']
plt.plot(x, y, label='VeryHighPen-DR3', color='blue', linestyle='-')
plt.fill_between(x, ymin, ymax, color='blue', alpha=0.2)

# Plot MedPen-HoleNoise (dashed line)
if not df['MedPen-HoleNoise - rollout/ep_len_mean'].isnull().all():
    y2 = df['MedPen-HoleNoise - rollout/ep_len_mean']
    ymin2 = df['MedPen-HoleNoise - rollout/ep_len_mean__MIN']
    ymax2 = df['MedPen-HoleNoise - rollout/ep_len_mean__MAX']
    plt.plot(x, y2, label='MedPen-HoleNoise', color='green', linestyle='-')
    plt.fill_between(x, ymin2, ymax2, color='green', alpha=0.2)

# Plot formatting
plt.xlabel('Training Steps')
plt.ylabel('Episode Length Mean')
# plt.title('Episode Length Learning Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save to PDF
plt.savefig("episode_length_curve.pdf")

# Show the plot
plt.show()