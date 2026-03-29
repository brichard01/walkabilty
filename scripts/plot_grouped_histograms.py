#!/usr/bin/env python
import os
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_all_ratings(base_dir, participants):
    """
    Build a DataFrame with columns [Participant, Segment, Rating].
    Each participant must have segments_{PID}.yaml under base_dir/PID/.
    """
    rows = []
    for pid in participants:
        yaml_path = os.path.join(base_dir, pid, f"segments_{pid}.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML not found for {pid}: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            doc = yaml.safe_load(f)
        for seg in doc['segments']:
            rows.append({
                "Participant": pid,
                "Segment": seg["segment_id"],
                "Rating": seg["walkability_rating"]
            })
    return pd.DataFrame(rows)

def plot_grouped_histograms(df, output_path):
    """
    Creates one subplot per participant showing a bar for each segment.
    """
    participants = df['Participant'].unique()
    n = len(participants)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    
    # If only one participant, axes is not a list
    if n == 1:
        axes = [axes]
    
    for ax, pid in zip(axes, participants):
        p_df = df[df["Participant"] == pid].sort_values("Segment")
        ax.bar(p_df["Segment"], p_df["Rating"], color="skyblue", edgecolor="gray")
        ax.set_title(f"{pid} · Walkability Ratings per Segment", fontsize=12)
        ax.set_ylabel("Rating", fontsize=10)
        ax.set_ylim(0, 5)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel("Segment ID", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved grouped histogram to {output_path}")

def main():
    p = argparse.ArgumentParser(
        description="Grouped bar plots of walkability ratings per segment for each participant"
    )
    p.add_argument(
        "--base_dir", required=True,
        help="Root data folder, e.g. C:/Abderrahim Internship Data/Data"
    )
    p.add_argument(
        "--participants", required=True, nargs="+",
        help="List of participant IDs, e.g. P01 P02 P03"
    )
    args = p.parse_args()
    
    df = load_all_ratings(args.base_dir, args.participants)
    
    plots_dir = os.path.join(args.base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_file = os.path.join(plots_dir, "all_participants_histograms.png")
    
    plot_grouped_histograms(df, output_file)

if __name__ == "__main__":
    main()
