import pandas as pd
import os

# === ✅ CONFIGURATION ===
participant_id = "P09"
base_dir = "C:/Abderrahim Internship Data/Data"
input_csv = os.path.join(base_dir, participant_id, f"{participant_id}_gaze_with_segment.csv")
output_dir = os.path.join(base_dir, participant_id, "objective_analysis")
os.makedirs(output_dir, exist_ok=True)

# ✅ Class labels mapping
class_labels = {
    0: "Bird", 1: "Ground Animal", 2: "Curb", 3: "Fence", 4: "Guard Rail", 5: "Barrier", 6: "Wall",
    7: "Bike Lane", 8: "Crosswalk - Plain", 9: "Curb Cut", 10: "Parking", 11: "Pedestrian Area",
    12: "Rail Track", 13: "Road", 14: "Service Lane", 15: "Sidewalk", 16: "Bridge", 17: "Building",
    18: "Tunnel", 19: "Person", 20: "Bicyclist", 21: "Motorcyclist", 22: "Other Rider",
    23: "Lane Marking - Crosswalk", 24: "Lane Marking - General", 25: "Mountain", 26: "Sand",
    27: "Sky", 28: "Snow", 29: "Terrain", 30: "Vegetation", 31: "Water", 32: "Banner", 33: "Bench",
    34: "Bike Rack", 35: "Billboard", 36: "Catch Basin", 37: "CCTV Camera", 38: "Fire Hydrant",
    39: "Junction Box", 40: "Mailbox", 41: "Manhole", 42: "Phone Booth", 43: "Pothole",
    44: "Street Light", 45: "Pole", 46: "Traffic Sign Frame", 47: "Utility Pole",
    48: "Traffic Light", 49: "Traffic Sign (Back)", 50: "Traffic Sign (Front)", 51: "Trash Can",
    52: "Bicycle", 53: "Boat", 54: "Bus", 55: "Car", 56: "Caravan", 57: "Motorcycle",
    58: "On Rails", 59: "Other Vehicle", 60: "Trailer", 61: "Truck", 62: "Wheeled Slow",
    63: "Car Mount", 64: "Ego Vehicle"
}

irrelevant_ids = [25, 26, 28, 53]

# === ✅ Load data
df = pd.read_csv(input_csv)
print(f"✅ Loaded: {len(df)} fixations")

# ✅ Filter out irrelevant classes
df = df[~df['class_id'].isin(irrelevant_ids)]

# ✅ Group by segment and class
grouped = df.groupby(['segment_id', 'class_id'])['fixation_duration_sec'].sum().reset_index()

# ✅ Map class IDs to labels
grouped['class_label'] = grouped['class_id'].map(class_labels)

# ✅ Pivot to wide format: segment_id as rows, classes as columns
pivot_duration = grouped.pivot(index='segment_id', columns='class_label', values='fixation_duration_sec').fillna(0)

# ✅ Save raw fixation durations
pivot_duration.to_csv(os.path.join(output_dir, f"{participant_id}_segment_fixation_duration.csv"))
print("✅ Saved: fixation duration per class per segment")

# ✅ Normalize to percentages per segment
percentages = pivot_duration.div(pivot_duration.sum(axis=1), axis=0) * 100
percentages = percentages.round(2)

# ✅ Save percentages
percentages.to_csv(os.path.join(output_dir, f"{participant_id}_segment_fixation_percentage.csv"))
print("✅ Saved: % of fixation time per class per segment")

# ✅ Done
print("\n🏁✅ Objective segment-level analysis completed!")
