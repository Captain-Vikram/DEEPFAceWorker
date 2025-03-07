import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Load the CSV file
csv_file = "emotion_results.csv"
df = pd.read_csv(csv_file)

# Convert frame numbers to time (assuming 30 FPS, adjust if needed)
fps = 30  # Change according to your video FPS
df["Time (s)"] = df["Frame"] / fps

# Smooth the confidence levels using Gaussian filter
df["Smoothed Confidence"] = gaussian_filter1d(df["Confidence(%)"], sigma=2)

# Plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
plt.plot(df["Time (s)"], df["Smoothed Confidence"], color="blue", linewidth=2, label="Confidence Level")

plt.xlabel("Time (seconds)")
plt.ylabel("Confidence Level (%)")
plt.title("Smoothed Confidence Level Over Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Save the plot as an image
plt.savefig("confidence_trend_smooth.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
