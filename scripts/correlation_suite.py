#!/usr/bin/env python
import os
import re
import ast
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def clean_np_float64_list(val):
    if isinstance(val, str):
        # strip out np.float64(...) wrappers
        cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', val)
        return ast.literal_eval(cleaned)
    return val

def main():
    # ─── CLI ─────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Generate correlation heatmap & scatter plots between gaze fixations and walkability"
    )
    parser.add_argument(
        "--participant_id", "-p",
        required=True,
        help="Participant ID, e.g. P01, P02, …"
    )
    args = parser.parse_args()
    pid = args.participant_id

    # ─── PATHS ────────────────────────────────────────────────────────────
    base_dir   = r"C:/Abderrahim Internship Data/Data"
    in_csv     = os.path.join(base_dir, pid, f"{pid}_final_analysis.csv")
    plot_folder= os.path.join(base_dir, pid, "plots")
    os.makedirs(plot_folder, exist_ok=True)

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Final analysis CSV not found: {in_csv}")

    # ─── LOAD & PARSE ─────────────────────────────────────────────────────
    df = pd.read_csv(in_csv)
    # parse the list‐columns
    df['top_classes']     = df['top_classes'].apply(ast.literal_eval)
    df['top_percentages'] = df['top_percentages'].apply(clean_np_float64_list)

    # ─── BUILD FIXATION MATRIX ───────────────────────────────────────────
    # all unique classes appearing in any segment
    all_classes = sorted({c for row in df['top_classes'] for c in row})
    # pivot table: rows=segment, cols=class, values=fixation%
    fix_df = pd.DataFrame(0.0, index=df['segment_id'].unique(), columns=all_classes)
    for _, row in df.iterrows():
        sid = row['segment_id']
        for cls, pct in zip(row['top_classes'], row['top_percentages']):
            fix_df.at[sid, cls] = pct
    # attach walkability
    fix_df['walkability'] = df.set_index('segment_id')['walkability_rating']

    # ─── 1️⃣ HEATMAP ──────────────────────────────────────────────────────
    corr = fix_df.corr().loc[all_classes, ['walkability']]
    plt.figure(figsize=(4, len(all_classes)*0.3 + 1))
    sns.heatmap(corr, annot=True, cmap="coolwarm", cbar_kws={'label':'Pearson r'})
    plt.title(f"{pid} – Corr(Fixation %, Walkability)")
    plt.xlabel("Walkability")
    plt.tight_layout()
    heatmap_path = os.path.join(plot_folder, f"{pid}_corr_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"✅ Heatmap saved → {heatmap_path}")

    # ─── 2️⃣ SCATTER + TRENDLINE ─────────────────────────────────────────
    # pick top 3 absolute correlations
    top3 = corr['walkability'].abs().sort_values(ascending=False).head(3).index.tolist()
    for cls in top3:
        plt.figure(figsize=(5,4))
        sns.regplot(
            x=fix_df[cls], y=fix_df['walkability'],
            scatter_kws={'s':60, 'alpha':0.7},
            line_kws={'color':'red'}
        )
        plt.title(f"{pid} – Fix% on {cls}\nvs Walkability")
        plt.xlabel(f"Fixation % on {cls}")
        plt.ylabel("Walkability Rating")
        plt.ylim(0,5)
        plt.tight_layout()
        scatter_path = os.path.join(plot_folder, f"{pid}_scatter_{cls.replace(' ','_')}.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"✅ Scatter saved → {scatter_path}")

if __name__=="__main__":
    main()
