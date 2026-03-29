#!/usr/bin/env python
import argparse
import glob
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_ratings(data_dir):
    """Scan Data/P*/segments_*.yaml and return a list of all walkability_ratings."""
    pattern = os.path.join(data_dir, "P*", "segments_*.yaml")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No segment YAMLs found with pattern {pattern}")
    ratings = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for seg in cfg.get("segments", []):
            ratings.append(seg["walkability_rating"])
    return ratings

def main():
    p = argparse.ArgumentParser(
        description="Plot pooled histogram of all segment walkability ratings"
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Root data directory, e.g. C:/Abderrahim Internship Data/Data"
    )
    p.add_argument(
        "--output", default=None,
        help="Path to output PNG (default: <data_dir>/plots/all_ratings_hist.png)"
    )
    args = p.parse_args()

    data_dir = args.data_dir
    out_path = args.output or os.path.join(data_dir, "plots", "all_ratings_hist.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load
    ratings = load_ratings(data_dir)
    df = pd.DataFrame({"rating": ratings})

    # Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8,5))
    sns.histplot(df["rating"], bins=20, kde=False, color="C2")
    plt.title("Histogram of All Segment Walkability Ratings")
    plt.xlabel("Walkability Rating (1–5)")
    plt.ylabel("Count of Segments")
    plt.xlim(1,5)
    plt.tight_layout()

    # Save
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved histogram to {out_path}")

if __name__ == "__main__":
    main()
