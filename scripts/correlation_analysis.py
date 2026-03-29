#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re

def clean_np_float64(val):
    """
    Remove wrappers like np.float64(12.3) in CSV strings.
    """
    if isinstance(val, str):
        s = re.sub(r'np\.float64\(([^\)]+)\)', r'\1', val)
        return ast.literal_eval(s)
    return val

def main():
    p = argparse.ArgumentParser(
        description="Compute & plot correlations between gaze fixation % and walkability"
    )
    p.add_argument("--participant_id", required=True,
                   help="e.g. P01, P02, …")
    args = p.parse_args()
    pid = args.participant_id

    # ─── Paths ────────────────────────────────────────────────────────
    base = r"C:/Abderrahim Internship Data/Data"
    fin_csv = os.path.join(base, pid, f"{pid}_final_analysis.csv")
    out_csv = os.path.join(base, pid, f"{pid}_corr_summary.csv")
    plot_dir = os.path.join(base, pid, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ─── Load merged analysis ─────────────────────────────────────────
    if not os.path.exists(fin_csv):
        raise FileNotFoundError(f"Missing final analysis: {fin_csv}")
    df = pd.read_csv(fin_csv)

    # ─── Parse list columns ───────────────────────────────────────────
    df['top_classes']     = df['top_classes'].apply(ast.literal_eval)
    df['top_percentages'] = df['top_percentages'].apply(clean_np_float64)

    # ─── Build fixation % pivot: rows=segment, cols=all classes ────────
    # gather all classes
    all_classes = sorted({c for lst in df['top_classes'] for c in lst})
    fix_df = pd.DataFrame(0.0, index=df['segment_id'], columns=all_classes)

    for _, row in df.iterrows():
        seg = row['segment_id']
        for cls, pct in zip(row['top_classes'], row['top_percentages']):
            fix_df.at[seg, cls] = pct

    # add subjective rating
    subj = df.set_index('segment_id')['walkability_rating']
    fix_df['walkability_rating'] = subj

    # ─── Compute Pearson r’s ──────────────────────────────────────────
    corr = fix_df.corr(method='pearson')['walkability_rating']\
                  .drop('walkability_rating')\
                  .rename("r")\
                  .reset_index()\
                  .rename(columns={'index':'class_label'})
    corr.to_csv(out_csv, index=False)
    print(f"✅ Saved correlation table to {out_csv}")

    # ─── 1) Heatmap of all r’s ────────────────────────────────────────
    plt.figure(figsize=(4, len(corr)*0.25+1))
    corr_plot = corr.sort_values('r', ascending=False).set_index('class_label')
    sns.heatmap(corr_plot, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label':'Pearson r'})
    plt.title(f"{pid} – Corr(Fixation %, Walkability)")
    plt.tight_layout()
    hmfile = os.path.join(plot_dir, f"{pid}_corr_heatmap.png")
    plt.savefig(hmfile); plt.close()
    print(f"✅ Saved heatmap to {hmfile}")

    # ─── 2) Scatter + trend for top and bottom 3 ───────────────────────
    top3 = corr.nlargest(3, 'r')['class_label'].tolist()
    bot3 = corr.nsmallest(3, 'r')['class_label'].tolist()
    for cls in top3 + bot3:
        plt.figure(figsize=(5,4))
        sns.regplot(x=fix_df[cls], y=fix_df['walkability_rating'],
                    ci=None, scatter_kws={'s':60}, line_kws={'color':'red'})
        plt.xlabel(f"Fixation % on {cls}")
        plt.ylabel("Walkability Rating")
        sign = "pos" if cls in top3 else "neg"
        plt.title(f"{pid}: {cls} ({'+' if cls in top3 else '-'}corr)")
        sp = os.path.join(plot_dir, f"{pid}_scatter_{sign}_{cls}.png")
        plt.tight_layout()
        plt.savefig(sp); plt.close()
        print(f"✅ Saved scatter for {cls} → {sp}")

if __name__=="__main__":
    main()
