import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import os

# ✅ SETTINGS
participant_id = "P01"
num_samples = 5  # Pick how many random fixations you want to check
base_dir = "C:/Abderrahim Internship Data/Data"
csv_file = os.path.join(base_dir, participant_id, f"{participant_id}_gaze_synced_real_with_class.csv")
frame_dir = os.path.join(base_dir, participant_id, "semantic_segmentation", "snapshots")
mask_dir = os.path.join(base_dir, participant_id, "semantic_segmentation", "masks")
# ✅ Load gaze data
df = pd.read_csv(csv_file)

# ✅ Filter valid fixations only (with real class IDs)
df = df[df['class_id'] >= 0]

# ✅ Randomly sample fixations
samples = df.sample(n=num_samples, random_state=35)

print(f"\n🎯 Checking {num_samples} random fixations from {participant_id}...\n")

for idx, row in samples.iterrows():
    frame_idx = int(row['frame_idx'])
    x = int(row['fixation_x'])
    y = int(row['fixation_y'])
    class_id = int(row['class_id'])

    # ✅ Load snapshot and mask
    frame_path = os.path.join(frame_dir, f"frame_{frame_idx:05d}.jpg")
    mask_path = os.path.join(mask_dir, f"mask_{frame_idx:05d}.png")

    if not os.path.exists(frame_path) or not os.path.exists(mask_path):
        print(f"⚠️ Skipping frame {frame_idx} — Missing image or mask")
        continue

    frame = np.array(Image.open(frame_path))
    mask = np.array(Image.open(mask_path))

    # ✅ Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.scatter(x, y, color='red', s=50)
    plt.title(f"📷 Frame {frame_idx} — Gaze at ({x},{y})")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='tab20')
    plt.scatter(x, y, color='red', s=50)
    plt.title(f"🟦 Mask {frame_idx} — Class ID: {class_id}")

    plt.tight_layout()
    plt.show()
