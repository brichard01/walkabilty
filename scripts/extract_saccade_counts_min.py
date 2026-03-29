#!/usr/bin/env python
import argparse, os, yaml
import pandas as pd
import numpy as np

DEFAULT_BASE = r"C:/Abderrahim Internship Data/Data"

# ---------- helpers ----------
def hhmmss_to_s(x):
    """Convert 'HH:MM:SS' or 'MM:SS' to seconds."""
    s = str(x).strip()
    if not s:
        return np.nan
    parts = [float(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)  # pad at the front
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return 3600*h + 60*m + sec

def load_segments(yml_path):
    with open(yml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    segs = []
    for seg in data.get("segments", []):
        segs.append({
            "segment_id": int(seg["segment_id"]),
            "start_s": hhmmss_to_s(seg["start_time"]),
            "end_s":   hhmmss_to_s(seg["end_time"]),
        })
    segs.sort(key=lambda d: d["segment_id"])
    return segs

def time_to_segment(t, segs):
    if pd.isna(t):
        return np.nan
    for s in segs:
        if s["start_s"] <= t < s["end_s"]:
            return s["segment_id"]
    return np.nan

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Count saccades per segment from Tobii TSV.")
    ap.add_argument("--participant_id", required=True, help="e.g., P03")
    ap.add_argument("--base_dir", default=DEFAULT_BASE)
    ap.add_argument("--sep", default="\t", help="TSV separator (default: tab)")
    # If your headers differ, you can override:
    ap.add_argument("--type_col", default="Eye movement type")
    ap.add_argument("--time_col", default="Recording timestamp")
    ap.add_argument("--dur_col",  default="Eye movement event duration")
    args = ap.parse_args()

    pid = args.participant_id
    base = args.base_dir

    # Locate TSV in raw/
    raw_dir = os.path.join(base, pid, "raw")
    tsv_path = None
    if os.path.isdir(raw_dir):
        for fn in os.listdir(raw_dir):
            if fn.lower().endswith(".tsv"):
                tsv_path = os.path.join(raw_dir, fn)
                break
    if not tsv_path:
        raise FileNotFoundError(f"No TSV found in {raw_dir}")

    # Load segments YAML
    yml = os.path.join(base, pid, f"segments_{pid}.yaml")
    if not os.path.exists(yml):
        raise FileNotFoundError(f"Missing {yml}")
    segs = load_segments(yml)
    seg_df = pd.DataFrame(segs)
    seg_df["duration_s"] = seg_df["end_s"] - seg_df["start_s"]

    # Read TSV as strings (robust), then convert what we need
    df = pd.read_csv(tsv_path, sep=args.sep, dtype=str, low_memory=False)

    # Sanity: ensure expected columns are present
    for col in [args.type_col, args.time_col]:
        if col not in df.columns:
            raise RuntimeError(f"Column not found in TSV: '{col}'")

    # Filter saccades
    typ = df[args.type_col].astype(str).str.lower()
    sacc = df[typ.str.contains("saccade", na=False)].copy()
    if sacc.empty:
        raise RuntimeError("No saccade rows found. Check --type_col or file contents.")

    # Convert time (Recording timestamp) → seconds (ms -> s)
    sacc["time_ms"] = pd.to_numeric(sacc[args.time_col], errors="coerce")
    sacc["time_s"]  = sacc["time_ms"] / 1000.0

    # Optional: saccade duration if present
    if args.dur_col in sacc.columns:
        sacc["dur_ms"] = pd.to_numeric(sacc[args.dur_col], errors="coerce")
        sacc["dur_s"]  = sacc["dur_ms"] / 1000.0
    else:
        sacc["dur_ms"] = np.nan
        sacc["dur_s"]  = np.nan

    # Map to segment
    sacc["segment_id"] = sacc["time_s"].apply(lambda t: time_to_segment(t, segs)).astype("Int64")

    # Export per-event times (for any later deep dives)
    out_dir = os.path.join(base, pid, "saccades")
    os.makedirs(out_dir, exist_ok=True)
    events_csv = os.path.join(out_dir, f"{pid}_saccades_times_with_segment.csv")
    sacc[["segment_id", "time_s", "dur_s"]].to_csv(events_csv, index=False)
    print(f"✅ per-saccade times saved: {events_csv}  (rows={len(sacc)})")

    # Aggregate per segment: count, total duration, mean duration, rate
    agg = (sacc.dropna(subset=["segment_id"])
                 .groupby("segment_id", as_index=False)
                 .agg(saccade_count=("segment_id", "size"),
                      sacc_total_dur_s=("dur_s", "sum"),
                      sacc_mean_dur_s=("dur_s", "mean")))

    # Merge with segment durations to compute rates
    summary = seg_df.merge(agg, on="segment_id", how="left")
    summary["saccade_count"]    = summary["saccade_count"].fillna(0).astype(int)
    summary["sacc_total_dur_s"] = summary["sacc_total_dur_s"].fillna(0.0)
    summary["sacc_mean_dur_s"]  = summary["sacc_mean_dur_s"]
    summary["saccades_per_sec"] = summary["saccade_count"] / summary["duration_s"].replace(0, np.nan)

    summary = summary[["segment_id", "duration_s",
                       "saccade_count", "saccades_per_sec",
                       "sacc_total_dur_s", "sacc_mean_dur_s"]]

    out_summary = os.path.join(out_dir, f"{pid}_saccade_counts_by_segment.csv")
    summary.to_csv(out_summary, index=False)
    print(f"✅ per-segment counts saved: {out_summary}")

    print("\nColumns used:")
    print(f"  type_col = {args.type_col}")
    print(f"  time_col = {args.time_col}  (ms → s)")
    print(f"  dur_col  = {args.dur_col}   (ms → s, if present)")

if __name__ == "__main__":
    main()
