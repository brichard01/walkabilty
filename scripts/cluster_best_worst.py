#!/usr/bin/env python
import os, argparse, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_subjective_from_yaml(base_dir, pid):
    ypath = os.path.join(base_dir, pid, f"segments_{pid}.yaml")
    with open(ypad := ypath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rows = []
    for seg in data["segments"]:
        rows.append({
            "participant_id": pid,
            "segment_id": int(seg["segment_id"]),
            "walkability_rating": float(seg["walkability_rating"])
        })
    return pd.DataFrame(rows)

def load_objective_percent(base_dir, pid):
    fpath = os.path.join(base_dir, pid, "objective_analysis", f"{pid}_segment_fixation_percentage.csv")
    df = pd.read_csv(fpath)
    # Ensure segment_id exists and is int
    if "segment_id" not in df.columns:
        df.rename(columns={df.columns[0]: "segment_id"}, inplace=True)
    df["segment_id"] = df["segment_id"].astype(int)
    # Drop junk columns if any
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df.set_index("segment_id")

def main():
    ap = argparse.ArgumentParser(description="Best vs Worst segment clustering across participants")
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--participants", nargs="+", required=True)
    ap.add_argument("--topn", type=int, default=10, help="Top-N classes by absolute difference to plot")
    args = ap.parse_args()

    per_participant = []   # rows: participant_id, best_seg, best_rating, worst_seg, worst_rating
    best_list = []         # list of DataFrames (class % for best segs)
    worst_list = []        # list of DataFrames (class % for worst segs)

    for pid in args.participants:
        # Subjective
        sub = load_subjective_from_yaml(args.base_dir, pid)
        # Pick best/worst by rating
        best_row = sub.loc[sub["walkability_rating"].idxmax()]
        worst_row = sub.loc[sub["walkability_rating"].idxmin()]

        per_participant.append({
            "participant_id": pid,
            "best_segment": int(best_row["segment_id"]),
            "best_rating": float(best_row["walkability_rating"]),
            "worst_segment": int(worst_row["segment_id"]),
            "worst_rating": float(worst_row["walkability_rating"]),
        })

        # Objective % per class
        obj = load_objective_percent(args.base_dir, pid)
        # Guard: ensure segments exist
        if best_row["segment_id"] in obj.index:
            best_list.append(obj.loc[[best_row["segment_id"]]])
        if worst_row["segment_id"] in obj.index:
            worst_list.append(obj.loc[[worst_row["segment_id"]]])

    per_df = pd.DataFrame(per_participant)
    out_dir = os.path.join(args.base_dir, "GLOBAL")
    os.makedirs(out_dir, exist_ok=True)
    per_df.to_csv(os.path.join(out_dir, "per_participant_best_worst.csv"), index=False)

    # Aggregate clusters
    if not best_list or not worst_list:
        print("No data found to aggregate. Check inputs.")
        return

    best_all = pd.concat(best_list, axis=0)   # rows are segments; cols are classes
    worst_all = pd.concat(worst_list, axis=0)

    best_mean = best_all.mean(axis=0).sort_values(ascending=False)
    worst_mean = worst_all.mean(axis=0).sort_values(ascending=False)

    best_mean.to_csv(os.path.join(out_dir, "best_cluster_mean.csv"), header=["mean_fixation_percent"])
    worst_mean.to_csv(os.path.join(out_dir, "worst_cluster_mean.csv"), header=["mean_fixation_percent"])

    # Choose top classes by absolute difference
    common_cols = best_mean.index.intersection(worst_mean.index)
    diff = (best_mean[common_cols] - worst_mean[common_cols]).abs().sort_values(ascending=False)
    top = diff.head(args.topn).index

    # Plot side-by-side bars for top classes
    x = np.arange(len(top))
    width = 0.42
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, best_mean[top].values, width, label="BEST (mean %)")
    plt.bar(x + width/2, worst_mean[top].values, width, label="WORST (mean %)")
    plt.xticks(x, top, rotation=45, ha="right")
    plt.ylabel("Mean fixation time (%)")
    plt.title("Best vs Worst Segments (Mean % per visual class)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"best_vs_worst_top{args.topn}.png"), dpi=200)
    plt.close()

    print(f"✅ Saved:\n  {os.path.join(out_dir,'per_participant_best_worst.csv')}\n  {os.path.join(out_dir,'best_cluster_mean.csv')}\n  {os.path.join(out_dir,'worst_cluster_mean.csv')}\n  {os.path.join(out_dir,f'best_vs_worst_top{args.topn}.png')}")

if __name__ == "__main__":
    main()
