import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Load the CSV file
csv_file = "emotion_results.csv"
df = pd.read_csv(csv_file)

# Ensure column names match your file
if 'Confidence (%)' in df.columns:
    confidence_col = 'Confidence (%)'
elif 'Confidence(%)' in df.columns:
    confidence_col = 'Confidence(%)'
else:
    confidence_cols = [col for col in df.columns if 'Confidence' in col]
    confidence_col = confidence_cols[0] if confidence_cols else 'Confidence'

# Convert frame numbers to time (assuming 30 FPS)
fps = 30
df["Time (s)"] = df["Frame"] / fps

# Sort data by time
df = df.sort_values("Time (s)")

# Create a figure with appropriate size
plt.figure(figsize=(14, 8))

# Apply heavy Gaussian smoothing (sigma=15)
sigma = 15
smoothed_confidence = gaussian_filter1d(df[confidence_col].values, sigma=sigma)

# Plot heavy smoothing
plt.plot(df["Time (s)"], smoothed_confidence, 
         color='#d62728', linewidth=2, label='Heavy Smoothing')

# Compute and plot rolling average
window_size = 15  # Adjust as needed
rolling_avg = df[confidence_col].rolling(window=window_size, center=True).mean()
plt.plot(df["Time (s)"], rolling_avg, 
         color='#9467bd', linewidth=2, linestyle='--', 
         label=f'Rolling Average (window={window_size})')

# Add styling and labels
plt.xlabel("Time (seconds)", fontsize=14)
plt.ylabel("Confidence Level (%)", fontsize=14)
plt.title("Smoothed Confidence Level Over Time", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc='lower right')

# Set y-axis limits
y_min = max(0, df[confidence_col].min() - 5)
y_max = min(100, df[confidence_col].max() + 5)
plt.ylim(y_min, y_max)

# Save the plot as an image
plt.tight_layout()
plt.savefig("confidence_analysis.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
