#!/usr/bin/env python
import argparse, os, yaml, ast, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def read_segments_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    rows = []
    for s in y.get("segments", []):
        rows.append({
            "segment_id": int(s["segment_id"]),
            "walkability_rating": float(s["walkability_rating"]),
            "feedback": str(s.get("feedback","")).strip(),
            "themes": list(s.get("themes", [])),
        })
    return pd.DataFrame(rows)

def load_final(path):
    df = pd.read_csv(path)
    # parse lists
    def parse_list(x):
        if isinstance(x, list): return x
        s = str(x)
        try:
            return ast.literal_eval(s)
        except:
            return [t.strip() for t in s.strip("[]").split(",")]
    df["top_classes"] = df["top_classes"].apply(parse_list)
    def clean_np_list(x):
        if isinstance(x, list): return [float(v) for v in x]
        s = str(x).replace("np.float64(","").replace(")","")
        return [float(v) for v in ast.literal_eval(s)]
    df["top_percentages"] = df["top_percentages"].apply(clean_np_list)
    return df

def main():
    ap = argparse.ArgumentParser(description="Aggregate ratings and best/worst across participants")
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--participants", nargs="+", required=True)   # e.g. P01 P02 P03 …
    args = ap.parse_args()

    out_dir = os.path.join(args.base_dir, "_aggregate")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Collect ratings (from segments_{PID}.yaml)
    ratings_rows = []
    for pid in args.participants:
        seg_yaml = os.path.join(args.base_dir, pid, f"segments_{pid}.yaml")
        df = read_segments_yaml(seg_yaml)
        df["participant_id"] = pid
        ratings_rows.append(df[["participant_id","segment_id","walkability_rating"]])
    all_ratings = pd.concat(ratings_rows, ignore_index=True)
    all_ratings.to_csv(os.path.join(out_dir, "all_ratings.csv"), index=False)

    # Plot grouped histograms (one small panel per participant)
    pids = args.participants
    n = len(pids)
    fig_h = max(2.0 * n, 8)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True)
    if n == 1: axes = [axes]
    for ax, pid in zip(axes, pids):
        sub = all_ratings[all_ratings["participant_id"]==pid]
        ax.bar(sub["segment_id"], sub["walkability_rating"], width=0.8, color="#8ecae6", edgecolor="#1d3557")
        ax.set_ylim(0,5); ax.set_ylabel("Rating"); ax.set_title(f"{pid} – Walkability Ratings per Segment")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xlabel("Segment ID")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ratings_grouped_histograms.png"), dpi=200)
    plt.close()

    # 2) Best vs Worst class averages across participants (using *_final_analysis.csv)
    per_pid_best = []
    per_pid_worst = []
    for pid in pids:
        # subjective to find best/worst
        seg_yaml = os.path.join(args.base_dir, pid, f"segments_{pid}.yaml")
        subj = read_segments_yaml(seg_yaml).sort_values(["walkability_rating","segment_id"], ascending=[False, True])
        best_seg = int(subj.iloc[0]["segment_id"])
        worst_seg = int(subj.sort_values(["walkability_rating","segment_id"], ascending=[True, True]).iloc[0]["segment_id"])

        # from final analysis get the top lists for those segments
        df_final = load_final(os.path.join(args.base_dir, pid, f"{pid}_final_analysis.csv"))
        brow = df_final[df_final["segment_id"]==best_seg].iloc[0]
        wrow = df_final[df_final["segment_id"]==worst_seg].iloc[0]

        per_pid_best.append(pd.Series(dict(zip(brow["top_classes"], brow["top_percentages"]))))
        per_pid_worst.append(pd.Series(dict(zip(wrow["top_classes"], wrow["top_percentages"]))))

    best_df = pd.DataFrame(per_pid_best).fillna(0.0)
    worst_df = pd.DataFrame(per_pid_worst).fillna(0.0)

    # align columns, keep common classes
    common_cols = sorted(set(best_df.columns) | set(worst_df.columns))
    best_df = best_df.reindex(columns=common_cols, fill_value=0.0)
    worst_df = worst_df.reindex(columns=common_cols, fill_value=0.0)

    avg_best = best_df.mean(axis=0).sort_values(ascending=False)
    avg_worst = worst_df.mean(axis=0).sort_values(ascending=False)

    summary = pd.DataFrame({
        "class": common_cols,
        "avg_fix_best": [avg_best.get(c,0.0) for c in common_cols],
        "avg_fix_worst": [avg_worst.get(c,0.0) for c in common_cols],
        "diff_best_minus_worst": [avg_best.get(c,0.0)-avg_worst.get(c,0.0) for c in common_cols],
    }).sort_values("diff_best_minus_worst", ascending=False)
    summary.to_csv(os.path.join(out_dir, "best_worst_summary.csv"), index=False)

    # Plot top 12 by |difference|
    top12 = summary.reindex(summary["diff_best_minus_worst"].abs().sort_values(ascending=False).index).head(12)
    x = np.arange(len(top12))
    w = 0.42
    fig = plt.figure(figsize=(14,6))
    plt.bar(x - w/2, top12["avg_fix_best"],  width=w, label="Best segments",  color="#219ebc", edgecolor="#1d3557")
    plt.bar(x + w/2, top12["avg_fix_worst"], width=w, label="Worst segments", color="#ffb703", edgecolor="#1d3557")
    plt.xticks(x, top12["class"], rotation=45, ha="right")
    plt.ylabel("Avg Fixation %")
    plt.title("Across participants: objective classes (Best vs Worst segments)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_vs_worst_topclasses.png"), dpi=200)
    plt.close()

    print(f"✅ Saved aggregate outputs to: {out_dir}")

if __name__ == "__main__":
    main()
