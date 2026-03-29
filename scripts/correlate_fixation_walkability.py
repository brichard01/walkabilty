#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def main():
    p = argparse.ArgumentParser(
        description="Compute & visualize correlation between gaze fixation % and walkability"
    )
    p.add_argument("--participant_id", required=True,
                   help="Participant ID, e.g. P01, P02, …")
    p.add_argument("--base_dir", default="C:/Abderrahim Internship Data/Data",
                   help="Root data folder")
    args = p.parse_args()
    pid = args.participant_id
    base = args.base_dir

    # paths
    gaze_pct = os.path.join(base, pid, "objective_analysis",
                            f"{pid}_segment_fixation_percentage.csv")
    subj_csv = os.path.join(base, pid, "subjective", f"{pid}_subjective.csv")
    corr_dir = os.path.join(base, pid, "objective_analysis")
    plot_dir = os.path.join(base, pid, "plots")
    os.makedirs(corr_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # load
    if not os.path.exists(gaze_pct):
        raise FileNotFoundError(gaze_pct)
    if not os.path.exists(subj_csv):
        raise FileNotFoundError(subj_csv)

    df_fix = pd.read_csv(gaze_pct, index_col=0)  # segments × classes
    df_sub = pd.read_csv(subj_csv).set_index("segment_id")

    # align segments
    common = df_fix.index.intersection(df_sub.index)
    df_fix = df_fix.loc[common]
    df_sub = df_sub.loc[common]

    # filter classes that appear (>0%) in at least 3 segments
    nonzero = (df_fix > 0).sum()
    keep = nonzero[nonzero >= 3].index.tolist()
    df_fix = df_fix[keep]

    # attach walkability
    df_fix["walkability"] = df_sub["walkability_rating"]

    # compute correlations
    records = []
    for cls in keep:
        x = df_fix[cls].values
        y = df_fix["walkability"].values
        r, p = pearsonr(x, y)
        n_nonzero = int((x > 0).sum())
        records.append({
            "class_label": cls,
            "r": r,
            "p_value": p,
            "n_nonzero": n_nonzero
        })
    corr_df = pd.DataFrame(records).set_index("class_label").sort_values("r", ascending=False)
    corr_csv = os.path.join(corr_dir, f"{pid}_corr_table.csv")
    corr_df.to_csv(corr_csv)
    print(f"✅ Saved correlation table → {corr_csv}")

    # heatmap of r
    plt.figure(figsize=(4, len(corr_df)*0.4 + 1.5))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    ax = sns.heatmap(corr_df[["r"]], annot=True, cmap=cmap, center=0,
                     cbar_kws={"label":"Pearson r"}, linewidths=0.5)
    # draw boxes around significant
    for i, cls in enumerate(corr_df.index):
        if corr_df.loc[cls, "p_value"] < 0.05:
            ax.add_patch(plt.Rectangle((0, i), 1, 1, fill=False, edgecolor="black", lw=2))
    ax.set_title(f"{pid} – Corr(Fix % vs Walkability)")
    ax.set_ylabel("")
    plt.tight_layout()
    heatmap_png = os.path.join(plot_dir, f"{pid}_corr_heatmap.png")
    plt.savefig(heatmap_png)
    plt.close()
    print(f"✅ Saved heatmap → {heatmap_png}")

    # scatter plots for top 3 absolute & significant
    sig = corr_df[corr_df.p_value < 0.05]
    top3 = sig["r"].abs().sort_values(ascending=False).head(3).index
    for cls in top3:
        plt.figure(figsize=(5,4))
        sns.regplot(x=df_fix[cls], y=df_fix["walkability"], ci=95,
                    scatter_kws={"s":60, "alpha":0.7},
                    line_kws={"color":"red", "lw":2})
        r = corr_df.loc[cls, "r"]
        pval = corr_df.loc[cls, "p_value"]
        plt.title(f"{pid}: {cls}\nr={r:.2f}, p={pval:.3f}")
        plt.xlabel(f"Fixation % on {cls}")
        plt.ylabel("Walkability Rating")
        plt.tight_layout()
        out = os.path.join(plot_dir, f"{pid}_scatter_{cls}.png")
        plt.savefig(out)
        plt.close()
        print(f"✅ Saved scatter → {out}")

if __name__ == "__main__":
    main()
