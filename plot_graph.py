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
    original_confidence_col = 'Confidence (%)'
elif 'Confidence(%)' in df.columns:
    original_confidence_col = 'Confidence(%)'
else:
    confidence_cols = [col for col in df.columns if 'Confidence' in col]
    original_confidence_col = confidence_cols[0] if confidence_cols else 'Confidence'

# Convert frame numbers to time (assuming 30 FPS)
fps = 30
df["Time (s)"] = df["Frame"] / fps

# Define confidence and stress levels
confidence_modifiers = {
    'neutral': 0.7, 'happy': 0.9, 'surprise': 0.5, 'fear': 0.3,
    'angry': 0.6, 'sad': 0.4, 'disgust': 0.5
}

stress_weights = {
    'neutral': 0.2, 'happy': 0.1, 'surprise': 0.6, 'fear': 0.9,
    'angry': 0.85, 'sad': 0.7, 'disgust': 0.75
}

def calculate_custom_confidence(row):
    return ((confidence_modifiers.get(row['Emotion'], 0.5) * 0.7) + 
            (row[original_confidence_col] / 100 * 0.3)) * 100

def calculate_stress(row):
    return stress_weights.get(row['Emotion'], 0.5) * row[original_confidence_col]

df['Custom Confidence'] = df.apply(calculate_custom_confidence, axis=1)
df['Stress Level'] = df.apply(calculate_stress, axis=1)

# Sort by time
df = df.sort_values("Time (s)")

# Apply smoothing
sigma = 15
smoothed_confidence = gaussian_filter1d(df['Custom Confidence'].values, sigma=sigma)
smoothed_stress = gaussian_filter1d(df['Stress Level'].values, sigma=12)

# Plot both confidence and stress
fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()

ax1.plot(df["Time (s)"], smoothed_confidence, color='#d62728', linewidth=2, label='Custom Confidence (Smoothed)')
ax2.plot(df["Time (s)"], smoothed_stress, color='#ff7f0e', linewidth=2, linestyle='--', label='Stress Level (Smoothed)')
ax2.fill_between(df["Time (s)"], 0, smoothed_stress, color='#ff7f0e', alpha=0.3)

# High stress threshold
high_stress_threshold = 60
ax2.axhline(y=high_stress_threshold, color='red', linestyle='--', alpha=0.7, label='High Stress Threshold')

# Add emotion labels on x-axis
emotion_changes = df[['Time (s)', 'Emotion']].drop_duplicates(subset=['Emotion'])
ax1.set_xticks(emotion_changes['Time (s)'])
ax1.set_xticklabels(emotion_changes['Emotion'], rotation=45, ha='right', fontsize=10, color='black')

# Labels and legends
ax1.set_xlabel("Time (seconds)", fontsize=14)
ax1.set_ylabel("Confidence Level (%)", fontsize=14, color='#d62728')
ax2.set_ylabel("Stress Level (%)", fontsize=14, color='#ff7f0e')
ax1.set_title("Estimated Confidence and Stress Levels Over Time", fontsize=16)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Save and show
plt.tight_layout()
plt.savefig("confidence_stress_analysis.png", dpi=300, bbox_inches="tight")
plt.show()