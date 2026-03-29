import numpy as np
import cv2
import os
from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm

# --------------------------
# Mapping Mask2Former
# --------------------------
id2label = {
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

irrelevant_ids = [0, 25, 26, 28, 53]  # skip these classes

# --------------------------
# Utility functions
# --------------------------
def time_to_seconds(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def mask_to_percentages(mask):
    """Return a dict with class proportions (0-1)"""
    mask_flat = mask.flatten()
    counts = np.bincount(mask_flat)
    total = mask_flat.size
    props = {}
    for class_id, name in id2label.items():
        if class_id in irrelevant_ids:
            continue
        props[name] = counts[class_id] / total if class_id < len(counts) else 0
    return props

# --------------------------
# IVW computation
# --------------------------
def calculate_IVW_levels(data):
    # Extract features
    tree = data.get('Vegetation', 0)
    building = data.get('Building', 0)
    pavement = data.get('Sidewalk', 0) + data.get('Pedestrian Area', 0)
    road = data.get('Road', 0)
    car = data.get('Car', 0) + data.get('Truck', 0) + data.get('Bus', 0)
    fence = data.get('Fence', 0)
    obstacles = data.get('Pole',0) + data.get('Traffic Sign (Back)',0) + \
                data.get('Traffic Sign (Front)',0) + data.get('Parking',0) + data.get('Billboard',0)

    # G
    G_i = tree
    if G_i < 0.07: G = 1
    elif G_i < 0.127: G = 2
    elif G_i < 0.175: G = 3
    elif G_i < 0.228: G = 4
    else: G = 5

    # C
    C_i = obstacles + car
    if C_i >= 0.09: C = 1
    elif C_i >= 0.044: C = 2
    elif C_i >= 0.02: C = 3
    elif C_i >= 0.007: C = 4
    else: C = 5

    # S
    if (pavement + road + fence) == 0:
        S = 1
    else:
        S_i = (building + tree) / (pavement + road + fence)
        if S_i < 0.8 or S_i >= 2.864: S = 1
        elif S_i < 1.063 or S_i >= 2.21: S = 2
        elif S_i < 1.213 or S_i >= 1.884: S = 3
        elif S_i < 1.35 or S_i >= 1.656: S = 4
        else: S = 5

    # D
    if road == 0:
        D = 1
    else:
        D_i = (pavement + fence) / road
        if D_i < 0.48 or D_i >= 4: D = 1
        elif D_i < 0.65 or D_i >= 3.007: D = 2
        elif D_i < 0.806 or D_i >= 2.137: D = 3
        elif D_i < 0.953 or D_i >= 1.707: D = 4
        else: D = 5

    return G, C, S, D

def compute_IVW_score(data):
    G, C, S, D = calculate_IVW_levels(data)
    return 5 * (G + C + S + D)

# --------------------------
# Process a segment
# --------------------------
def compute_segment_pv_ivw(mask_folder, start_frame, end_frame):
    ivw_list = []
    accumulated = {}
    n_frames = 0

    for frame_idx in range(start_frame, end_frame + 1):
        mask_path = Path(mask_folder) / f"mask_{frame_idx:06d}.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ Skipping unreadable file: {mask_path}")
            continue

        data = mask_to_percentages(mask)

        # Method 1
        ivw_list.append(compute_IVW_score(data))

        # Method 2 accumulation
        for k,v in data.items():
            accumulated[k] = accumulated.get(k,0) + v

        n_frames += 1

    if n_frames == 0:
        return None, None

    # Method 1: average IVW
    m1 = np.mean(ivw_list)

    # Method 2: compute IVW from averaged proportions
    avg_data = {k: v/n_frames for k,v in accumulated.items()}
    m2 = compute_IVW_score(avg_data)

    return m1, m2

# --------------------------
# Process all segments from YAML
# --------------------------
def process_segments(yaml_path, mask_folder, fps):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    results = []
    for seg in tqdm(data["segments"], desc="Segments"):
        start_frame = int(time_to_seconds(seg["start_time"]) * fps)
        end_frame   = int(time_to_seconds(seg["end_time"]) * fps)
        m1, m2 = compute_segment_pv_ivw(mask_folder, start_frame, end_frame)
        results.append({
            "segment_id": seg["segment_id"],
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "PV_IVW_M1": m1,
            "PV_IVW_M2": m2
        })
    return pd.DataFrame(results)

# --------------------------
# Usage example
# --------------------------
participant_id = "P09" \
""
base_dir = r"D:\Abderrahim Internship Data\Data"

yaml_path = os.path.join(base_dir, participant_id, f"new_segments_{participant_id}.yaml")
mask_folder = os.path.join(base_dir, participant_id, "semantic_segmentation", "masks")
fps = 24.95

df_ivw = process_segments(yaml_path, mask_folder, fps)
df_ivw.to_csv(os.path.join(base_dir, participant_id, f"pv_ivw_{participant_id}.csv"), index=False)
print(df_ivw)