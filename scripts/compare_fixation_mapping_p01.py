import pandas as pd
import os

# === ✅ SETTINGS
participant_id = "P01"
ROOT_DIR = r"C:\Abderrahim Internship Data\Data"
tsv_path = os.path.join(ROOT_DIR, participant_id, "raw", f"{participant_id}.tsv")
mapped_csv = os.path.join(ROOT_DIR, participant_id, f"{participant_id}_gaze_with_class.csv")
fps = 25  # Adjust to real FPS if needed

# === ✅ Load both files
print("🔄 Loading files...")
raw_df = pd.read_csv(tsv_path, sep="\t")
mapped_df = pd.read_csv(mapped_csv)
print(f"📄 Raw TSV: {len(raw_df)} rows | Mapped CSV: {len(mapped_df)} rows")

# ✅ OPTIONAL: Show mapped CSV columns to confirm structure
print("\n🧠 Mapped CSV columns:", mapped_df.columns.tolist())

# === ✅ Keep fixations only in TSV
raw_fixations = raw_df[raw_df['Eye movement type'] == 'Fixation'].copy()

# === ✅ Compute frame index from timestamp in TSV
frame_duration_us = (1 / fps) * 1_000_000
raw_fixations['frame_idx'] = (raw_fixations['Recording timestamp'] / frame_duration_us).astype(int)
raw_fixations = raw_fixations.rename(columns={
    'Fixation point X': 'fixation_x',
    'Fixation point Y': 'fixation_y',
    'Recording timestamp': 'timestamp_us'
})

# === ✅ Ask for a frame index
target_frame = int(input("\n🔍 Enter frame index to inspect: "))

# === ✅ Find fixations at that frame
fix_raw = raw_fixations[raw_fixations['frame_idx'] == target_frame]
fix_mapped = mapped_df[mapped_df['frame_idx'] == target_frame]

# === ✅ Show comparison
print(f"\n🎯 Fixations at frame {target_frame}:\n")

print("📄 From raw TSV:")
if fix_raw.empty:
    print("❌ No fixations in raw TSV for this frame.")
else:
    for i, row in fix_raw.iterrows():
        print(f"  • Time: {int(row['timestamp_us'])} μs | Pos: ({int(row['fixation_x'])}, {int(row['fixation_y'])})")

print("\n📄 From mapped CSV:")
if fix_mapped.empty:
    print("❌ No mapped fixations for this frame.")
else:
    for i, row in fix_mapped.iterrows():
        print(f"  • Time: {int(row['timestamp_us'])} μs | Pos: ({int(row['fixation_x'])}, {int(row['fixation_y'])})"
              f" → Class ID: {row['class_id']} | Label: {row.get('class_label', 'N/A')}")
