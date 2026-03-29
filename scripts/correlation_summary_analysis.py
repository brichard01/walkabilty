# correlation_summary_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from matplotlib.offsetbox import AnchoredText
import re


# === 🔧 CONFIG
participant_id = "P01"
base_dir = "C:/Abderrahim Internship Data/Data"
input_csv = os.path.join(base_dir, participant_id, f"{participant_id}_final_analysis.csv")
output_dir = os.path.join(base_dir, participant_id, "plots")
os.makedirs(output_dir, exist_ok=True)

# === 📥 Load and clean data
df = pd.read_csv(input_csv)
df['top_classes'] = df['top_classes'].apply(ast.literal_eval)
df['top_percentages'] = df['top_percentages'].apply(lambda s: ast.literal_eval(re.sub(r'np\.float64\((.*?)\)', r'\1', s)))
df['themes'] = df['themes'].apply(ast.literal_eval)

# === 📊 Prepare expanded DataFrame for plotting
rows = []
for _, row in df.iterrows():
    for cls, pct in zip(row['top_classes'], row['top_percentages']):
        rows.append({
            'segment_id': row['segment_id'],
            'walkability_rating': row['walkability_rating'],
            'class_label': cls,
            'percentage': pct,
            'themes': ", ".join(row['themes']),
            'feedback': row['feedback']
        })

df_exp = pd.DataFrame(rows)

# === 🎨 Plot: Gaze Fixation + Walkability Rating
plt.figure(figsize=(14, 7))
sns.barplot(data=df_exp, x="segment_id", y="percentage", hue="class_label", palette="Set2")
sns.lineplot(data=df_exp.drop_duplicates('segment_id'), x="segment_id", y="walkability_rating",
             color="black", marker="o", label="Walkability", linewidth=2)

plt.title(f"{participant_id} – Gaze Fixation vs Walkability Rating", fontsize=14)
plt.ylabel("Fixation Time (%)")
plt.xlabel("Segment")
plt.ylim(0, 60)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# === 🗒️ Annotate with feedback below plot
for i, row in df.iterrows():
    feedback = row['feedback']
    themes = ", ".join(row['themes'])
    note = f"★ {feedback}\nThemes: {themes}"
    plt.gca().annotate(note, xy=(row['segment_id']-1, -5), xycoords='data',
                       fontsize=8, ha='center', va='top', rotation=0)

# ✅ Save figure
plot_path = os.path.join(output_dir, f"{participant_id}_gaze_walkability_corr.png")
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {plot_path}")

# === 📄 Create summary table
summary = []
for _, row in df.iterrows():
    top_str = ", ".join([f"{c} ({p:.1f}%)" for c, p in zip(row['top_classes'], row['top_percentages'])])
    summary.append({
        "Segment": row['segment_id'],
        "Walkability Rating": row['walkability_rating'],
        "Top Visual Elements": top_str,
        "Themes": ", ".join(row['themes']),
        "Feedback Summary": row['feedback']
    })

df_summary = pd.DataFrame(summary)
summary_path = os.path.join(output_dir, f"{participant_id}_correlation_summary.csv")
df_summary.to_csv(summary_path, index=False)
print(f"✅ Saved: {summary_path}")
