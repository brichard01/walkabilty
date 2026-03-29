#!/usr/bin/env python
import os, argparse, yaml, ast
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_BASE = r"C:/Abderrahim Internship Data/Data"

def drop_unnamed(df):
    return df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")].copy()

def load_subjective(base_dir, pid):
    subj_csv = os.path.join(base_dir, pid, "subjective", f"{pid}_subjective.csv")
    if os.path.exists(subj_csv):
        df = pd.read_csv(subj_csv)
        df = drop_unnamed(df)
        df["segment_id"] = df["segment_id"].astype(int)
        df["walkability_rating"] = df["walkability_rating"].astype(float)
        return df[["segment_id","walkability_rating"]].copy()

    # Fallback: parse YAML
    yml = os.path.join(base_dir, pid, f"segments_{pid}.yaml")
    with open(yml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rows = []
    for seg in data.get("segments", []):
        rows.append({"segment_id": int(seg["segment_id"]),
                     "walkability_rating": float(seg["walkability_rating"])})
    return pd.DataFrame(rows)

def load_saccades(base_dir, pid):
    path = os.path.join(base_dir, pid, "objective_analysis", f"{pid}_saccades_by_segment.csv")
    df = pd.read_csv(path)
    df = drop_unnamed(df)
    # ensure col names
    if "segment_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "segment_id"})
    df["segment_id"] = df["segment_id"].astype(int)
    return df

def main():
    ap = argparse.ArgumentParser(description="Per-participant saccade-rate vs walkability plot")
    ap.add_argument("--participant_id", required=True)
    ap.add_argument("--base_dir", default=DEFAULT_BASE)
    args = ap.parse_args()

    pid = args.participant_id
    subj = load_subjective(args.base_dir, pid)
    sacc = load_saccades(args.base_dir, pid)
    # saccades_per_sec column should exist from extractor; if not, compute
    if "saccades_per_sec" not in sacc.columns and {"saccade_count","duration_s"}.issubset(sacc.columns):
        sacc["saccades_per_sec"] = sacc["saccade_count"] / sacc["duration_s"].replace({0: pd.NA})

    df = pd.merge(sacc, subj, on="segment_id", how="inner").sort_values("segment_id")
    out_dir = os.path.join(args.base_dir, pid, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{pid}_saccades_rate_vs_rating.png")

    # Plot: bars (saccades/sec) + line (rating)
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.bar(df["segment_id"], df["saccades_per_sec"], width=0.6, edgecolor="black")
    ax1.set_xlabel("Segment")
    ax1.set_ylabel("Saccades / sec")
    ax1.set_xticks(df["segment_id"])

    ax2 = ax1.twinx()
    ax2.plot(df["segment_id"], df["walkability_rating"], color="crimson", marker="o", linewidth=2, label="Rating")
    ax2.set_ylim(0,5)
    ax2.set_ylabel("Walkability rating (1–5)")

    plt.title(f"{pid} – Saccade rate per segment vs Walkability")
    fig.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"✅ Saved: {out_png}")

if __name__ == "__main__":
    main()
