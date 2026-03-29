#!/usr/bin/env python
import os
import argparse
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    p = argparse.ArgumentParser(
        description="Plot frequency of subjective themes per segment"
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
    csv_path  = os.path.join(base, "subjective", f"{PID}_subjective.csv")
    plots_dir = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_png   = os.path.join(plots_dir, f"{PID}_subjective_themes.png")

    # load & explode themes
    df = pd.read_csv(csv_path)
    # themes column is stored like "['a','b',...]" → parse with ast.literal_eval
    df['themes'] = df['themes'].apply(ast.literal_eval)
    df_ex = df.explode('themes').rename(columns={'themes':'theme'})

    # count
    counts = ( df_ex
        .groupby(['segment_id','theme'])
        .size()
        .reset_index(name='count')
    )

    # plot
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=counts,
        x="count", y="theme",
        hue="segment_id",
        dodge=True
    )
    plt.title(f"{PID} — Themes Mentioned per Segment", fontsize=14)
    plt.xlabel("Mentions")
    plt.ylabel("Theme")
    plt.legend(title="Segment ID", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()

    # save
    plt.savefig(out_png)
    plt.close()
    print(f"✅ Saved theme plot → {out_png}")

if __name__=="__main__":
    main()
