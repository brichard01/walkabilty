# generate_segment_correlation_summary.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import re

# === 🔧 CONFIGURATION ===
participant_id = "P01"
base_dir       = "C:/Abderrahim Internship Data/Data"
input_csv      = os.path.join(base_dir, participant_id, f"{participant_id}_final_analysis.csv")
output_dir     = os.path.join(base_dir, participant_id, "plots", "segments")
os.makedirs(output_dir, exist_ok=True)

# === 📥 LOAD DATA ===
df = pd.read_csv(input_csv)

# === 🛠️ CLEAN & PARSE FIELDS ===
def clean_and_parse_list(raw_str):
    """
    1) Remove np.float64(...) wrappers
    2) literal_eval to convert to Python list
    """
    if not isinstance(raw_str, str):
        return raw_str
    # strip np.float64(…) calls
    cleaned = re.sub(r"np\.float64\(\s*(.*?)\s*\)", r"\1", raw_str)
    return ast.literal_eval(cleaned)

# top_classes: plain list of strings
df['top_classes'] = df['top_classes'].apply(ast.literal_eval)

# top_percentages: remove np.float64(...) then parse
df['top_percentages'] = df['top_percentages'].apply(clean_and_parse_list)

# themes: plain list of strings
df['themes'] = df['themes'].apply(ast.literal_eval)

# === 🔁 GENERATE ONE PLOT PER SEGMENT ===
sns.set(style="whitegrid")
for _, row in df.iterrows():
    seg_id      = row['segment_id']
    gaze_labels = row['top_classes']
    gaze_pcts   = row['top_percentages']
    rating      = row['walkability_rating']
    feedback    = row['feedback']
    themes_list = row['themes']
    
    # 📊 Barplot of top gaze fixations
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=gaze_labels, y=gaze_pcts, palette="Set2")
    
    # ⚫️ Overlay walkability as a dashed horizontal line:
    #    we scale rating (1–5) to 0–50% by multiplying by 10
    walk_y = rating * 10
    ax.axhline(walk_y, color="black", linestyle="--", linewidth=2,
               label=f"Walkability ×10 ({rating})")
    
    # 🎨 Style
    ax.set_ylim(0, max(gaze_pcts) * 1.2)
    ax.set_title(f"{participant_id} – Segment {seg_id} | Walkability: {rating}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Top Visual Elements", fontsize=12)
    ax.set_ylabel("Fixation Duration (%)", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    
    # 📝 Add feedback & themes below the plot
    themes_text = " • ".join(themes_list)
    plt.figtext(0.5, -0.12,
                f"📝 Feedback: {feedback}",
                wrap=True, ha="center", fontsize=9)
    plt.figtext(0.5, -0.18,
                f"🎯 Themes: {themes_text}",
                wrap=True, ha="center", fontsize=8, color="dimgray")
    
    ax.legend(loc="upper right")
    plt.tight_layout()
    
    # 💾 Save
    out_path = os.path.join(output_dir, f"{participant_id}_segment_{seg_id}_summary.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"✅ Saved: {out_path}")
