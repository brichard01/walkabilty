#!/usr/bin/env python
import argparse
import glob
import os
import yaml

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_all_ratings(data_dir):
    """
    Scan for Data/P*/segments_*.yaml,
    parse segment_id & walkability_rating,
    return concatenated DataFrame.
    """
    pattern = os.path.join(data_dir, "P*/segments_*.yaml")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No YAML files found with pattern {pattern}")

    records = []
    for path in files:
        pid = os.path.basename(os.path.dirname(path))  # e.g. "P08"
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for seg in cfg.get("segments", []):
            records.append({
                "participant_id": pid,
                "segment_id": seg["segment_id"],
                "walkability_rating": seg["walkability_rating"]
            })

    return pd.DataFrame.from_records(records)

def main():
    p = argparse.ArgumentParser(
        description="Plot distribution of walkability ratings across all participants/segments"
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Root data directory, e.g. C:/Abderrahim Internship Data/Data"
    )
    p.add_argument(
        "--output", default=None,
        help="Output PNG path. Defaults to <data_dir>/plots/ratings_distribution.png"
    )
    args = p.parse_args()

    data_dir = args.data_dir
    out_path = args.output or os.path.join(data_dir, "plots", "ratings_distribution.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ─── Load all ratings ───────────────────────────────────────────────
    df = load_all_ratings(data_dir)
    df["walkability_rating"] = pd.to_numeric(df["walkability_rating"], errors="coerce")

    # ─── Plot ───────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Left: overall histogram + KDE
    sns.histplot(
        df["walkability_rating"],
        bins=10, kde=True, color="C0", ax=axes[0]
    )
    axes[0].set_title("All Segments: Walkability Ratings")
    axes[0].set_xlabel("Rating (1–5)")
    axes[0].set_ylabel("Count")

    # Right: boxplot by participant
    order = (
        df.groupby("participant_id")["walkability_rating"]
          .median()
          .sort_values(ascending=False)
          .index
          .tolist()
    )
    sns.boxplot(
        data=df,
        x="participant_id", y="walkability_rating",
        order=order, palette="pastel", ax=axes[1]
    )
    axes[1].set_title("Ratings by Participant")
    axes[1].set_xlabel("Participant")
    axes[1].set_ylabel("Rating")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"✅ Saved plot to {out_path}")

if __name__ == "__main__":
    main()
