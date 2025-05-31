import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("reward_tuning.csv")

# Convert Step to numeric
df['Step'] = pd.to_numeric(df['Step'])

# Apply global font settings
plt.rcParams.update({
    'font.size': 16,           # Controls default text size
    'axes.labelsize': 16,      # Axis label size
    'xtick.labelsize': 12,     # X tick label size
    'ytick.labelsize': 12,     # Y tick label size
    'legend.fontsize': 14,     # Legend font size
})

# Create figure
plt.figure(figsize=(10, 6))

# Extract data
x = df['Step']

# Plot With Off-Edge Reset (blue)
y1 = df['With Off-Edge Reset - rollout/ep_rew_mean']
y1min = df['With Off-Edge Reset - rollout/ep_rew_mean__MIN']
y1max = df['With Off-Edge Reset - rollout/ep_rew_mean__MAX']
plt.plot(x, y1, label='With Off-Edge Reset', color='blue')
plt.fill_between(x, y1min, y1max, color='blue', alpha=0.2)

# Plot Without Off-Edge Reset (green)
y2 = df['Without Off-Edge Reset - rollout/ep_rew_mean']
y2min = df['Without Off-Edge Reset - rollout/ep_rew_mean__MIN']
y2max = df['Without Off-Edge Reset - rollout/ep_rew_mean__MAX']
plt.plot(x, y2, label='Without Off-Edge Reset', color='green')
plt.fill_between(x, y2min, y2max, color='green', alpha=0.2)

# Labels and layout
plt.xlabel('Training Steps')
plt.ylabel('Mean Episode Reward')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("reward_tuning.pdf", format="pdf", bbox_inches="tight")
plt.show()
