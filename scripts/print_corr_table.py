#!/usr/bin/env python
import argparse
import pandas as pd
import os

def main():
    p = argparse.ArgumentParser(
        description="Print correlation summary as Markdown"
    )
    p.add_argument(
        "--participant_id",
        required=True,
        help="Participant ID, e.g. P01, P02, …"
    )
    args = p.parse_args()
    pid = args.participant_id

    base = r"C:/Abderrahim Internship Data/Data"
    csv = os.path.join(base, pid, f"{pid}_corr_summary.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(f"No correlation file at {csv}")
    
    df = pd.read_csv(csv)
    # Only keep the top N if you like:
    # df = df.head(10)

    # Round r and p‐values for readability
    df["r"]       = df["r"].round(2)
    df["p_value"] = df["p_value"].apply(lambda v: f"{v:.3f}")
    
    # Output markdown
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
