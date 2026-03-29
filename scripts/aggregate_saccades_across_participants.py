#!/usr/bin/env python
import os, argparse, yaml, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_BASE = r"C:/Abderrahim Internship Data/Data"

def drop_unnamed(df): return df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")].copy()

def load_subjective(base, pid):
    subj_csv = os.path.join(base, pid, "subjective", f"{pid}_subjective.csv")
    if os.path.exists(subj_csv):
        df = pd.read_csv(subj_csv); df = drop_unnamed(df)
        df["segment_id"] = df["segment_id"].astype(int)
        df["walkability_rating"] = df["walkability_rating"].astype(float)
        return df[["segment_id","walkability_rating"]].copy()
    yml = os.path.join(base, pid, f"segments_{pid}.yaml")
    with open(yml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rows=[]
    for seg in data.get("segments", []):
        rows.append({"segment_id": int(seg["segment_id"]),
                     "walkability_rating": float(seg["walkability_rating"])})
    return pd.DataFrame(rows)

def load_saccades(base, pid):
    path = os.path.join(base, pid, "objective_analysis", f"{pid}_saccades_by_segment.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path); df = drop_unnamed(df)
    if "segment_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "segment_id"})
    df["segment_id"] = df["segment_id"].astype(int)
    if "saccades_per_sec" not in df.columns and {"saccade_count","duration_s"}.issubset(df.columns):
        df["saccades_per_sec"] = df["saccade_count"] / df["duration_s"].replace({0: np.nan})
    return df

def best_worst(subj_df):
    b = subj_df.sort_values(["walkability_rating","segment_id"], ascending=[False,True]).iloc[0]
    w = subj_df.sort_values(["walkability_rating","segment_id"], ascending=[True, True]).iloc[0]
    return int(b["segment_id"]), float(b["walkability_rating"]), int(w["segment_id"]), float(w["walkability_rating"])

def main():
    ap = argparse.ArgumentParser(description="Aggregate saccades + subjective across participants")
    ap.add_argument("--base_dir", default=DEFAULT_BASE, required=True)
    ap.add_argument("--participants", nargs="+", required=True, help="e.g. P01 P02 ...")
    args = ap.parse_args()

    rows = []
    bw_rows = []

    for pid in args.participants:
        subj = load_subjective(args.base_dir, pid)
        sacc = load_saccades(args.base_dir, pid)
        if sacc is None or sacc.empty:
            print(f"⚠️  Skip {pid}: no saccade file.")
            continue
        df = pd.merge(sacc, subj, on="segment_id", how="inner")
        df.insert(0, "participant_id", pid)
        rows.append(df[["participant_id","segment_id","duration_s","saccade_count","saccades_per_sec","walkability_rating"]])

        # best vs worst label
        b_id, b_r, w_id, w_r = best_worst(subj)
        df["bw_label"] = np.where(df["segment_id"]==b_id, "best",
                           np.where(df["segment_id"]==w_id, "worst", "other"))
        bw_rows.append(df[["participant_id","segment_id","saccades_per_sec","walkability_rating","bw_label"]])

    if not rows:
        raise RuntimeError("No data aggregated.")
    agg = pd.concat(rows, ignore_index=True)
    bw  = pd.concat(bw_rows, ignore_index=True)

    out_dir = os.path.join(args.base_dir, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    agg_csv = os.path.join(out_dir, "aggregate_saccades_subjective.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"✅ Saved: {agg_csv}")

    # Per-participant correlation
    corrs = []
    for pid, g in agg.groupby("participant_id"):
        if g["saccades_per_sec"].notna().sum() >= 2:
            r = g["saccades_per_sec"].corr(g["walkability_rating"])
        else:
            r = np.nan
        corrs.append({"participant_id": pid, "r_sacc_rate_vs_rating": r, "n_segments": len(g)})
    corr_df = pd.DataFrame(corrs).sort_values("participant_id")
    corr_csv = os.path.join(out_dir, "agg_corr_by_participant.csv")
    corr_df.to_csv(corr_csv, index=False)
    print(f"✅ Saved: {corr_csv}")

    # Global scatter rating vs saccades/sec
    plt.figure(figsize=(7.5,6))
    sns.regplot(data=agg, x="saccades_per_sec", y="walkability_rating", scatter_kws={"s":45, "alpha":0.8}, line_kws={"color":"crimson"})
    plt.xlabel("Saccades per second")
    plt.ylabel("Walkability rating (1–5)")
    plt.title("All participants – Rating vs Saccade rate")
    plt.tight_layout()
    out_scatter = os.path.join(out_dir, "agg_scatter_rating_vs_saccades.png")
    plt.savefig(out_scatter, dpi=220); plt.close()
    print(f"✅ Saved: {out_scatter}")

    # Best vs worst boxplot
    bw2 = bw[bw["bw_label"].isin(["best","worst"])].copy()
    plt.figure(figsize=(6.5,6))
    sns.boxplot(data=bw2, x="bw_label", y="saccades_per_sec")
    sns.stripplot(data=bw2, x="bw_label", y="saccades_per_sec", color="black", alpha=0.5, jitter=0.15)
    plt.xlabel("")
    plt.ylabel("Saccades per second")
    plt.title("Saccade rate: Best vs Worst segment (across participants)")
    plt.tight_layout()
    out_box = os.path.join(out_dir, "best_vs_worst_sacc_rate_box.png")
    plt.savefig(out_box, dpi=220); plt.close()
    print(f"✅ Saved: {out_box}")

if __name__ == "__main__":
    main()
