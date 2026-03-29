import yaml
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------
# Fonctions utilitaires
# --------------------------
def time_to_seconds(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def load_participant_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

# --------------------------
# Mapping Mask2Former
# --------------------------
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
irrelevant_ids = [0, 25, 26, 28, 53]

def compute_pixel_proportions_mask2former_named(mask):
    """Calcule les proportions pour toutes les classes présentes et donne leur nom"""
    mask_flat = mask.flatten()
    counts = np.bincount(mask_flat)
    total = mask_flat.size
    props = {}
    for class_id, name in class_labels.items():
        if class_id in irrelevant_ids:
            continue
        if class_id < len(counts):
            props[name] = counts[class_id] / total
        else:
            props[name] = 0
    return props

# --------------------------
# Traitement segment
# --------------------------
def process_segment(seg, fps, masks_dir):
    seg_id = seg["segment_id"]
    t_start = time_to_seconds(seg["start_time"])
    t_end = time_to_seconds(seg["end_time"])
    start_frame = int(t_start * fps)
    end_frame = int(t_end * fps)

    proportions_list = []

    for frame_idx in range(start_frame, end_frame + 1):
        mask_path = Path(masks_dir) / f"mask_{frame_idx:06d}.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        proportions_list.append(compute_pixel_proportions_mask2former_named(mask))

    if not proportions_list:
        return None

    df_props = pd.DataFrame(proportions_list).fillna(0)
    mean_props = df_props.mean()
    std_props = df_props.std()
    eps = 1e-6
    entropy = -np.sum(mean_props * np.log(mean_props + eps))

    row = {
        "segment_id": seg_id,
        "walkability_rating": seg.get("walkability_rating", np.nan),
        "entropy": entropy,
    }

    for c in mean_props.index:
        row[f"{c}_mean"] = mean_props[c]
        row[f"{c}_std"] = std_props[c]

    return row

# --------------------------
# Extraction features participant
# --------------------------
def extract_viewport_features_optimized(yaml_path, masks_dir, max_workers=8):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    data = load_participant_yaml(yaml_path)
    participant_id = data["participant_id"]
    fps = data["fps"]
    segments = data["segments"]

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_segment, seg, fps, masks_dir): seg for seg in segments}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Participant {participant_id}"):
            res = future.result()
            if res is not None:
                res["participant_id"] = participant_id
                results.append(res)

    return pd.DataFrame(results)

# --------------------------
# Paramètres et exécution
# --------------------------
participant_id = "P09" \
""
base_dir = r"D:\Abderrahim Internship Data\Data"

yaml_path = os.path.join(base_dir, participant_id, f"new_segments_{participant_id}.yaml")
masks_dir = os.path.join(base_dir, participant_id, "semantic_segmentation", "masks")
output_file = os.path.join(base_dir, participant_id, f"{participant_id}_viewport_features_optimised.csv")

df_viewport = extract_viewport_features_optimized(yaml_path, masks_dir, max_workers=12)
df_viewport.to_csv(output_file, index=False)
print(f"CSV enregistré : {output_file}")

