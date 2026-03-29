#!/usr/bin/env python3
import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant_id", required=True)
    parser.add_argument("--base_dir", default="C:/Abderrahim Internship Data/Data")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    csv_in = os.path.join(args.base_dir, args.participant_id,
                          "objective_analysis",
                          f"{args.participant_id}_segment_fixation_percentage.csv")
    out_dir = os.path.join(args.base_dir, args.participant_id, "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_in, index_col=0)
    # ── CLEAN ───────────────────────────────────────────────
    if -1 in df.index:
        df = df.drop(-1, axis=0)
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]

    # pick top-N classes by mean across segments
    top = df.mean().sort_values(ascending=False).head(args.top_n).index
    df_top = df[top]

    ax = df_top.plot(kind="bar", stacked=True, figsize=(10,5))
    ax.set_xlabel("Segment")
    ax.set_ylabel("Fixation Time (%)")
    ax.set_title(f"{args.participant_id} – Stacked gaze % (top {args.top_n} classes)")
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"{args.participant_id}_stacked_segments_top{args.top_n}.png")
    plt.savefig(out_png)
    plt.close()
    print(f"✅ Saved stacked bar chart to {out_png}")

if __name__=="__main__":
    main()
