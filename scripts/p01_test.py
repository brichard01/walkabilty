import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge
from matplotlib import cm
import numpy as np

# ✅ Load your data
csv_path = "../Data/P01/P01_gaze_synced_real_with_class_segmented.csv"
df = pd.read_csv(csv_path)

# ✅ Class label mapping
class_labels = {
    0: "Bird", 1: "Ground Animal", 2: "Curb", 3: "Fence", 4: "Guard Rail", 5: "Barrier", 6: "Wall",
    7: "Bike Lane", 8: "Crosswalk - Plain", 9: "Curb Cut", 10: "Parking", 11: "Pedestrian Area", 12: "Rail Track",
    13: "Road", 14: "Service Lane", 15: "Sidewalk", 16: "Bridge", 17: "Building", 18: "Tunnel",
    19: "Person", 20: "Bicyclist", 21: "Motorcyclist", 22: "Other Rider", 23: "Lane Marking - Crosswalk",
    24: "Lane Marking - General", 25: "Mountain", 26: "Sand", 27: "Sky", 28: "Snow", 29: "Terrain",
    30: "Vegetation", 31: "Water", 32: "Banner", 33: "Bench", 34: "Bike Rack", 35: "Billboard", 36: "Catch Basin",
    37: "CCTV Camera", 38: "Fire Hydrant", 39: "Junction Box", 40: "Mailbox", 41: "Manhole", 42: "Phone Booth",
    43: "Pothole", 44: "Street Light", 45: "Pole", 46: "Traffic Sign Frame", 47: "Utility Pole", 48: "Traffic Light",
    49: "Traffic Sign (Back)", 50: "Traffic Sign (Front)", 51: "Trash Can", 52: "Bicycle", 53: "Boat",
    54: "Bus", 55: "Car", 56: "Caravan", 57: "Motorcycle", 58: "On Rails", 59: "Other Vehicle",
    60: "Trailer", 61: "Truck", 62: "Wheeled Slow", 63: "Car Mount", 64: "Ego Vehicle"
}
irrelevant_ids = [25, 26, 28, 53]
df = df[~df['class_id'].isin(irrelevant_ids)]
df['class_label'] = df['class_id'].map(class_labels)

# ✅ Aggregate by segment and class
agg = df.groupby(['segment_id', 'class_label'])['fixation_duration_sec'].sum().reset_index()

# ✅ Normalize per segment to get %
agg['percentage'] = agg.groupby('segment_id')['fixation_duration_sec'].transform(lambda x: 100 * x / x.sum())

# ✅ Create pivot table for easier plotting
pivot = agg.pivot(index='segment_id', columns='class_label', values='percentage').fillna(0)

# ✅ Define segments order and colors
segments = pivot.index.tolist()
classes = pivot.columns.tolist()
colors = sns.color_palette("hls", len(classes))

# ✅ Plot nested donut (like in paper)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
theta = np.linspace(0.0, 2 * np.pi, len(classes), endpoint=False)
width = 2 * np.pi / len(classes)

# Each ring = a segment
for i, seg_id in enumerate(segments):
    radii = pivot.loc[seg_id].values
    inner_radius = 1 + i * 0.3
    bars = ax.bar(theta, radii, width=width, bottom=inner_radius, color=colors, edgecolor='white', linewidth=0.7)

# ✅ Add labels in the inner ring only
ax.set_xticks(theta)
ax.set_xticklabels(classes, fontsize=9, rotation=90)
ax.set_yticklabels([])
ax.set_title("🔍 Total Fixation Time per Visual Class (Segmented)", fontsize=14)
plt.tight_layout()
plt.savefig("P01_donut_plot.png", dpi=300)
plt.show()
