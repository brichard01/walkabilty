#!/usr/bin/env python
# aggregate_best_worst_across_participants.py
import argparse, os, ast, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DEFAULT_BASE = r"C:/Abderrahim Internship Data/Data"
IRRELEVANT = {"Mountain","Sand","Snow","Boat"}  # drop from analysis
JUNK_PREFIX = "unnamed"  # drop columns that start with this (case-insensitive)


# ---------- HELPERS ----------
def drop_junk_cols(df: pd.DataFrame) -> pd.DataFrame:
    junk = [c for c in df.columns if c.lower().startswith(JUNK_PREFIX)]
    return df.drop(columns=junk, errors="ignore")

def load_subjective(base_dir: str, pid: str) -> pd.DataFrame:
    """
    Load subjective ratings (segment_id, walkability_rating, feedback, themes)
    from either CSV (…/subjective/Pxx_subjective.csv) or segments YAML.
    """
    csv_path = os.path.join(base_dir, pid, "subjective", f"{pid}_subjective.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = drop_junk_cols(df)
        # ensure schema
        df["segment_id"] = df["segment_id"].astype(int)
        df["walkability_rating"] = df["walkability_rating"].astype(float)

        def parse_themes(x):
            if isinstance(x, list): return x
            s = str(x).strip()
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list): return [str(t).strip() for t in v]
            except Exception:
                pass
            return [t.strip() for t in s.split(",") if t.strip()]

        if "themes" in df.columns:
            df["themes"] = df["themes"].apply(parse_themes)
        else:
            df["themes"] = [[] for _ in range(len(df))]

        if "feedback" not in df.columns:
            df["feedback"] = ""

        return df[["segment_id","walkability_rating","feedback","themes"]].copy()

    # fallback to YAML
    yml_path = os.path.join(base_dir, pid, f"segments_{pid}.yaml")
    with open(yml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rows = []
    for seg in data.get("segments", []):
        rows.append({
            "segment_id": int(seg["segment_id"]),
            "walkability_rating": float(seg["walkability_rating"]),
            "feedback": str(seg.get("feedback","")).strip(),
            "themes": list(seg.get("themes", [])),
        })
    return pd.DataFrame(rows)

def load_objective_percent(base_dir: str, pid: str) -> pd.DataFrame:
    """
    Load per-segment fixation percentages by class:
      …/objective_analysis/Pxx_segment_fixation_percentage.csv
    Returns a DataFrame indexed by segment_id.
    """
    f = os.path.join(base_dir, pid, "objective_analysis", f"{pid}_segment_fixation_percentage.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Objective percentages CSV not found: {f}")

    df = pd.read_csv(f)
    df = drop_junk_cols(df)

    if "segment_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "segment_id"})
    df["segment_id"] = df["segment_id"].astype(int)
    df = df.set_index("segment_id")

    # drop irrelevant + junk
    keep = [c for c in df.columns if (c not in IRRELEVANT) and (not c.lower().startswith(JUNK_PREFIX))]
    df = df[keep].copy()

    # numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def best_worst_rows(subj: pd.DataFrame, obj_pct: pd.DataFrame):
    """
    Pick best (max rating) and worst (min rating) segment rows from objective df.
    If a segment is missing in objective df, return zeros for that row.
    """
    subj = subj.sort_values(["walkability_rating","segment_id"], ascending=[False, True])
    best = subj.iloc[0]
    subj_w = subj.sort_values(["walkability_rating","segment_id"], ascending=[True, True])
    worst = subj_w.iloc[0]

    def row_for(seg_id: int) -> pd.Series:
        if seg_id in obj_pct.index:
            return obj_pct.loc[seg_id]
        # zeros with same columns
        return pd.Series({c: 0.0 for c in obj_pct.columns})

    return (int(best["segment_id"]), row_for(int(best["segment_id"]))), \
           (int(worst["segment_id"]), row_for(int(worst["segment_id"])))

def safe_concat(series_list):
    """
    Concat a list of Series with possibly different columns -> align on union.
    """
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1).T
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


# ---------- PLOTS ----------
def plot_grouped_bar(top_classes, best_mean, worst_mean, outpath, title_suffix="Top classes (mean %)"):
    """
    Grouped bars for Best vs Worst on selected classes (list order kept).
    """
    classes = top_classes
    x = np.arange(len(classes))
    bw = 0.42

    plt.figure(figsize=(14, 6))
    plt.bar(x - bw/2, best_mean[classes], width=bw, label="Best segments", edgecolor='black')
    plt.bar(x + bw/2, worst_mean[classes], width=bw, label="Worst segments", edgecolor='black')

    plt.xticks(x, classes, rotation=45, ha='right')
    plt.ylabel("Mean fixation time (%)")
    plt.title(f"Best vs Worst – {title_suffix}")
    ymax = float(max(best_mean[classes].max(), worst_mean[classes].max()) * 1.2) if len(classes) else 10
    plt.ylim(0, ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()

def plot_diverging_diff(diff_series, topN, outpath, title="Best – Worst (mean % difference)"):
    """
    Diverging horizontal bars by absolute difference.
    """
    take = diff_series.reindex(diff_series.abs().sort_values(ascending=False).head(topN).index)
    classes = list(take.index)[::-1]
    vals = list(take.values)[::-1]

    plt.figure(figsize=(10, 8))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
    y = np.arange(len(classes))
    plt.barh(y, vals, color=colors, edgecolor='black')
    plt.yticks(y, classes)
    plt.axvline(0, color="#888", linewidth=1)
    for i, v in enumerate(vals):
        plt.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:+.1f}%", va='center',
                 ha='left' if v >= 0 else 'right', fontsize=9)
    plt.xlabel("Mean % (Best – Worst)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()


# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser(description="Aggregate best vs worst segments across participants")
    ap.add_argument("--base_dir", default=DEFAULT_BASE, help="Base data dir (contains P01, P02, …)")
    ap.add_argument("--participants", nargs="+", required=True, help="List like P01 P02 P03 …")
    ap.add_argument("--topN", type=int, default=12, help="How many classes to show in grouped bars")
    args = ap.parse_args()

    base = args.base_dir
    pids = args.participants
    out_dir = os.path.join(base, "_aggregate")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    best_rows = []
    worst_rows = []

    # Track which segments were used (for transparency)
    used_rows = []

    for pid in pids:
        try:
            subj = load_subjective(base, pid)
            objp = load_objective_percent(base, pid)
        except Exception as e:
            print(f"⚠️ Skipping {pid} due to load error: {e}")
            continue

        (best_id, best_row), (worst_id, worst_row) = best_worst_rows(subj, objp)

        # collect
        best_row.name = pid
        worst_row.name = pid
        best_rows.append(best_row)
        worst_rows.append(worst_row)

        used_rows.append({
            "participant": pid,
            "best_segment_id": best_id,
            "best_rating": float(subj.loc[subj["segment_id"] == best_id, "walkability_rating"].iloc[0]),
            "worst_segment_id": worst_id,
            "worst_rating": float(subj.loc[subj["segment_id"] == worst_id, "walkability_rating"].iloc[0]),
        })

    # align columns across participants
    best_df = safe_concat(best_rows)
    worst_df = safe_concat(worst_rows)

    if best_df.empty or worst_df.empty:
        raise RuntimeError("No aggregate data collected. Check inputs/participants.")

    # mean per class
    best_mean = best_df.mean(axis=0).sort_index()
    worst_mean = worst_df.mean(axis=0).sort_index()

    # export CSVs
    pd.DataFrame({"class": best_mean.index, "mean_fixation_pct": best_mean.values}) \
      .to_csv(os.path.join(out_dir, "aggregate_best_fixation_mean.csv"), index=False)
    pd.DataFrame({"class": worst_mean.index, "mean_fixation_pct": worst_mean.values}) \
      .to_csv(os.path.join(out_dir, "aggregate_worst_fixation_mean.csv"), index=False)
    pd.DataFrame(used_rows).to_csv(os.path.join(out_dir, "aggregate_used_segments.csv"), index=False)

    # choose TopN classes to show (by overall prominence)
    overall = ((best_mean + worst_mean) / 2.0).sort_values(ascending=False)
    top_classes = list(overall.head(args.topN).index)

    # grouped bars (Best vs Worst)
    plot_grouped_bar(
        top_classes, best_mean, worst_mean,
        outpath=os.path.join(plot_dir, f"best_vs_worst_grouped_top{args.topN}.png"),
        title_suffix=f"Top {args.topN} classes (mean % across participants)"
    )

    # diverging difference plot
    diff = (best_mean - worst_mean).sort_values()
    plot_diverging_diff(
        diff, topN=args.topN,
        outpath=os.path.join(plot_dir, f"best_minus_worst_diff_top{args.topN}.png"),
        title="Best – Worst (mean % difference) – Top classes by |diff|"
    )

    print("✅ Aggregation complete.")
    print(f"   • Best mean CSV:   {os.path.join(out_dir, 'aggregate_best_fixation_mean.csv')}")
    print(f"   • Worst mean CSV:  {os.path.join(out_dir, 'aggregate_worst_fixation_mean.csv')}")
    print(f"   • Used segments:   {os.path.join(out_dir, 'aggregate_used_segments.csv')}")
    print(f"   • Grouped bars:    {os.path.join(plot_dir, f'best_vs_worst_grouped_top{args.topN}.png')}")
    print(f"   • Diverging diff:  {os.path.join(plot_dir, f'best_minus_worst_diff_top{args.topN}.png')}")
    print("\nTip: include these two figures + the used-segments table in your slides for a clean story.")
    

if __name__ == "__main__":
    main()
