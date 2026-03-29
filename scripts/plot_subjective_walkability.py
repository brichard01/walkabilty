#!/usr/bin/env python
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    p = argparse.ArgumentParser(
        description="Plot walkability ratings per segment for a participant"
    )
    p.add_argument(
        "--participant_id", "-p",
        required=True,
        help="Participant folder name, e.g. P01, P02, etc."
    )
    args = p.parse_args()
    PID = args.participant_id

    # build paths
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data", PID))
    csv_path   = os.path.join(base, "subjective", f"{PID}_subjective.csv")
    plots_dir  = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_png    = os.path.join(plots_dir, f"{PID}_subjective_ratings.png")

    # load
    df = pd.read_csv(csv_path)

    # plot
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="segment_id", y="walkability_rating", palette="viridis")
    plt.title   (f"{PID} — Walkability Ratings per Segment", fontsize=14)
    plt.xlabel  ("Segment ID")
    plt.ylabel  ("Walkability Rating (1–5)")
    plt.ylim(0,5)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # save
    plt.savefig(out_png)
    plt.close()
    print(f"✅ Saved walkability plot → {out_png}")

if __name__=="__main__":
    main()
