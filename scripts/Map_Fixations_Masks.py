import pandas as pd
import numpy as np
import cv2
import os

# === ✅ CONFIGURATION ===
fps = 24.95
base_dir = "C:/Abderrahim Internship Data/Data"
participants = [f"P0{i}" for i in range(9, 10)]  # P01 to P09

for participant_id in participants:
    print(f"\n🔁 Processing {participant_id}...")

    # === ✅ Paths
    xlsx_file = os.path.join(base_dir, participant_id, "raw", f"{participant_id}.xlsx")
    mask_folder = os.path.join(base_dir, participant_id, "semantic_segmentation", "masks")
    output_csv = os.path.join(base_dir, participant_id, f"{participant_id}_gaze_synced_real_with_class.csv")

    # === ✅ Check if Excel exists
    if not os.path.exists(xlsx_file):
        print(f"❌ Missing XLSX for {participant_id} — skipping.")
        continue

    # === ✅ Load Excel and filter fixations
    df = pd.read_excel(xlsx_file)
    df = df[df['Eye movement type'] == 'Fixation'].copy()
    print(f"✅ Total fixations: {len(df)}")

    # === ✅ Rename columns (safe mapping for standard format)
    df = df.rename(columns={
        'Recording timestamp [μs]': 'timestamp_us',
        'Fixation point X [MCS px]': 'fixation_x',
        'Fixation point Y [MCS px]': 'fixation_y',
        'Eye movement event duration [ms]': 'fixation_duration_ms'
    })

    # === ✅ Compute frame index and duration
    df['frame_idx'] = (df['timestamp_us'] / 1_000_000 * fps).astype(int)
    df['fixation_duration_sec'] = df['fixation_duration_ms'] / 1000.0
    df['class_id'] = -1  # Default value

    # === ✅ Loop through fixations and map class
    for idx, row in df.iterrows():
        frame_idx = int(row['frame_idx'])
        x = int(row['fixation_x'])
        y = int(row['fixation_y'])

        mask_path = os.path.join(mask_folder, f"mask_{frame_idx:06d}.png")
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape
        if 0 <= x < w and 0 <= y < h:
            class_id = int(mask[y, x])
        else:
            class_id = -1

        df.at[idx, 'class_id'] = class_id

        if idx % 500 == 0:
            print(f"   🔁 {participant_id}: {idx}/{len(df)} fixations processed")

    # === ✅ Save result
    df[['frame_idx', 'timestamp_us', 'fixation_x', 'fixation_y', 'fixation_duration_sec', 'class_id']].to_csv(output_csv, index=False)
    print(f"✅ {participant_id} finished — saved to: {output_csv}")
