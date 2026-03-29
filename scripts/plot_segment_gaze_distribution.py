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
    # drop phantom segment
    if -1 in df.index:
        df = df.drop(-1, axis=0)
    # drop stray Unnamed columns
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]

    for seg in df.index:
        row = df.loc[seg].sort_values(ascending=False).head(args.top_n)
        plt.figure(figsize=(8,4))
        plt.barh(row.index, row.values)
        plt.gca().invert_yaxis()
        plt.title(f"{args.participant_id} Segment {seg} – Top {args.top_n} Classes by Gaze %")
        plt.xlabel("Fixation Time (%)")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{args.participant_id}_segment_{seg}_top{args.top_n}_gaze.png")
        plt.savefig(out_png)
        plt.close()
        print(f"✅ Saved plot for segment {seg} → {out_png}")

if __name__=="__main__":
    main()
