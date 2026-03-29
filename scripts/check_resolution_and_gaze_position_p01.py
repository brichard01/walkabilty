import cv2
from PIL import Image
import pandas as pd
import os

# === ✅ SETTINGS ===
participant_id = "P01"
frame_idx = 8479
root_dir = r"C:\Abderrahim Internship Data\Data"
video_path = os.path.join(root_dir, participant_id, "raw", "Video.mp4")
frame_path = os.path.join(root_dir, participant_id, "semantic_segmentation", "snapshots", f"frame_{frame_idx:05d}.jpg")
mask_path = os.path.join(root_dir, participant_id, "semantic_segmentation", "masks", f"mask_{frame_idx:05d}.png")
tsv_path = os.path.join(root_dir, participant_id, "raw", f"{participant_id}.tsv")
mapped_path = os.path.join(root_dir, participant_id, f"{participant_id}_gaze_with_class.csv")

print("🔍 Checking resolution and fixation data...\n")

# === ✅ 1. Load video resolution
cap = cv2.VideoCapture(video_path)
vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.release()
print(f"🎥 Video resolution: {int(vid_width)}x{int(vid_height)}")

# === ✅ 2. Load frame resolution
frame = Image.open(frame_path)
print(f"🖼️ Frame size: {frame.size[0]}x{frame.size[1]}")

# === ✅ 3. Load mask resolution
mask = Image.open(mask_path)
print(f"🌀 Mask size: {mask.size[0]}x{mask.size[1]}")

# === ✅ 4. Load raw TSV fixations
df_raw = pd.read_csv(tsv_path, sep="\t")
fps = 25
frame_duration_us = (1 / fps) * 1_000_000
df_raw['frame_idx'] = (df_raw['Recording timestamp'] / frame_duration_us).astype(int)

df_raw_fix = df_raw[(df_raw['Eye movement type'] == 'Fixation') & (df_raw['frame_idx'] == frame_idx)]
print(f"\n📌 {len(df_raw_fix)} raw fixations at frame {frame_idx}:")
for i, row in df_raw_fix.iterrows():
    print(f"  • Raw TSV fixation: ({int(row['Fixation point X'])}, {int(row['Fixation point Y'])})")

# === ✅ 5. Load mapped fixations
if os.path.exists(mapped_path):
    df_mapped = pd.read_csv(mapped_path)
    df_mapped_fix = df_mapped[df_mapped['frame_idx'] == frame_idx]
    print(f"\n📌 {len(df_mapped_fix)} mapped fixations at frame {frame_idx}:")
    for i, row in df_mapped_fix.iterrows():
        print(f"  • Mapped fixation: ({int(row['fixation_x'])}, {int(row['fixation_y'])}) → Class ID: {row['class_id']}")
else:
    print("\n⚠️ Mapped CSV not found.")
