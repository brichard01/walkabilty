#!/usr/bin/env python
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_means(base_dir):
    agg_dir = os.path.join(base_dir, "_aggregate")
    best_csv  = os.path.join(agg_dir, "aggregate_best_fixation_mean.csv")
    worst_csv = os.path.join(agg_dir, "aggregate_worst_fixation_mean.csv")

    if not os.path.exists(best_csv) or not os.path.exists(worst_csv):
        raise FileNotFoundError("Missing aggregate CSVs. Run aggregate_best_worst_across_participants.py first.")

    best  = pd.read_csv(best_csv)   .set_index("class")["mean_fixation_pct"]
    worst = pd.read_csv(worst_csv)  .set_index("class")["mean_fixation_pct"]

    # align classes across both
    all_idx = sorted(set(best.index) | set(worst.index))
    best  = best.reindex(all_idx).fillna(0.0)
    worst = worst.reindex(all_idx).fillna(0.0)
    return best, worst, agg_dir

def plot_grouped(best, worst, outpath, topK=15):
    overall = ((best + worst)/2.0).sort_values(ascending=False).head(topK)
    classes = list(overall.index)[::-1]  # reverse for nicer horizontal plot

    bvals = best.loc[classes].values
    wvals = worst.loc[classes].values

    y = np.arange(len(classes))
    h = 0.38

    plt.figure(figsize=(12, max(6, 0.45*len(classes)+2)))
    plt.barh(y - h/2, bvals, height=h, label="Best mean %", edgecolor='black')
    plt.barh(y + h/2, wvals, height=h, label="Worst mean %", edgecolor='black')
    plt.yticks(y, classes)
    plt.xlabel("Mean fixation time (%)")
    plt.title(f"Best vs Worst (Top {topK} classes by overall prominence)")
    xmax = float(max(bvals.max() if len(bvals) else 0, wvals.max() if len(wvals) else 0) * 1.25 or 10)
    plt.xlim(0, xmax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()

def plot_diverging(best, worst, outpath, topK=15):
    diff = (best - worst)  # positive => more in Best
    top = diff.reindex(diff.abs().sort_values(ascending=False).head(topK).index)
    classes = list(top.index)[::-1]
    vals = list(top.values)[::-1]

    plt.figure(figsize=(10, max(6, 0.45*len(classes)+2)))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
    y = np.arange(len(classes))
    plt.barh(y, vals, color=colors, edgecolor='black')
    plt.yticks(y, classes)
    plt.axvline(0, color="#888", linewidth=1)
    for i, v in enumerate(vals):
        plt.text(v + (0.4 if v>=0 else -0.4), i, f"{v:+.1f}%", va='center',
                 ha='left' if v>=0 else 'right', fontsize=9)
    plt.xlabel("Best – Worst (mean %)")
    plt.title(f"Drivers of Best/Worst (Top {topK} by |diff|)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Visualize aggregate CSVs (Best/Worst means)")
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--topK", type=int, default=15)
    args = ap.parse_args()

    best, worst, agg_dir = load_means(args.base_dir)
    plots_dir = os.path.join(agg_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    out_grouped  = os.path.join(plots_dir, f"aggregate_grouped_top{args.topK}.png")
    out_diverg   = os.path.join(plots_dir, f"aggregate_best_minus_worst_top{args.topK}.png")

    plot_grouped(best, worst, out_grouped, topK=args.topK)
    plot_diverging(best, worst, out_diverg, topK=args.topK)

    print("✅ Saved:")
    print(f"   • {out_grouped}")
    print(f"   • {out_diverg}")

if __name__ == "__main__":
    main()
