import pandas as pd
import numpy as np
import yaml
import os

# === ✅ CONFIGURATION
participant_id = "P09"
base_dir = "C:/Abderrahim Internship Data/Data"
fps = 24.95

# === ✅ PATHS
csv_file = os.path.join(base_dir, participant_id, f"{participant_id}_gaze_synced_real_with_class.csv")
yaml_file = os.path.join(base_dir, participant_id, "segments_P09.yaml")
output_file = os.path.join(base_dir, participant_id, f"{participant_id}_gaze_with_segment.csv")

# === ✅ Load gaze data
df = pd.read_csv(csv_file)
print(f"✅ Loaded: {len(df)} fixations")

# === ✅ Load YAML with segments
with open(yaml_file, "r") as f:
    config = yaml.safe_load(f)

segments = config["segments"]

# === ✅ Convert HH:MM:SS to frame index
def time_to_frame(time_str):
    h, m, s = map(int, time_str.split(":"))
    total_seconds = h * 3600 + m * 60 + s
    return int(total_seconds * fps)

# === ✅ Assign segment_id to each fixation
df['segment_id'] = -1

for segment in segments:
    segment_id = segment['segment_id']
    start_frame = time_to_frame(segment['start_time'])
    end_frame = time_to_frame(segment['end_time'])

    df.loc[(df['frame_idx'] >= start_frame) & (df['frame_idx'] < end_frame), 'segment_id'] = segment_id

print("✅ Assigned segment IDs")

# === ✅ Save new file
df.to_csv(output_file, index=False)
print(f"🏁✅ Done. Saved to {output_file}")
