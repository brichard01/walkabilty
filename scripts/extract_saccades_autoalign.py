#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract saccade metrics per segment with auto time alignment to YAML segments.

Outputs:
  <BASE>/<PID>/objective_analysis/<PID>_saccades_by_segment.csv

Requirements:
  pip install pandas numpy pyyaml

Typical usage:
  python extract_saccades_autoalign.py --participant_id P03 --base_dir "C:/Abderrahim Internship Data/Data" --debug
  # If you want to try the other clock:
  python extract_saccades_autoalign.py --participant_id P03 --base_dir "C:/Abderrahim Internship Data/Data" --time_col "Computer timestamp" --debug
"""

import os, glob, argparse, math
import numpy as np
import pandas as pd
import yaml

DEFAULT_BASE = r"C:/Abderrahim Internship Data/Data"

SACCADE_LABEL = "Saccade"
TYPE_COL = "Eye movement type"
IDX_COL  = "Eye movement type index"
DUR_COL  = "Eye movement event duration"
REC_DUR_COL = "Recording duration"       # helpful hint for unit guessing

def hhmmss_to_seconds(hhmmss: str) -> float:
    hh, mm, ss = hhmmss.strip().split(":")
    return int(hh)*3600 + int(mm)*60 + float(ss)

def find_segments_yaml(base_dir, pid):
    yml = os.path.join(base_dir, pid, f"segments_{pid}.yaml")
    if not os.path.exists(yml):
        raise FileNotFoundError(f"YAML not found: {yml}")
    with open(yml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    segs = []
    for seg in data.get("segments", []):
        if "start_time" not in seg or "end_time" not in seg:
            raise ValueError(f"segments_{pid}.yaml must include start_time/end_time for segment {seg.get('segment_id')}")
        segs.append({
            "segment_id": int(seg["segment_id"]),
            "start_s": hhmmss_to_seconds(seg["start_time"]),
            "end_s":   hhmmss_to_seconds(seg["end_time"])
        })
    segs = sorted(segs, key=lambda x: x["segment_id"])
    return pd.DataFrame(segs)

def find_tsv(base_dir, pid, override=None):
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"TSV override not found: {override}")
        return override
    # try /raw/*.tsv then /PID/*.tsv
    raw_dir = os.path.join(base_dir, pid, "raw")
    cands = glob.glob(os.path.join(raw_dir, "*.tsv"))
    if not cands:
        cands = glob.glob(os.path.join(base_dir, pid, "*.tsv"))
    if not cands:
        raise FileNotFoundError(f"No TSV found in {raw_dir} or {os.path.join(base_dir,pid)}")
    # pick the first
    return cands[0]

def choose_time_col(df, preferred=None):
    if preferred and preferred in df.columns:
        return preferred
    # fallbacks
    if "Recording timestamp" in df.columns:
        return "Recording timestamp"
    if "Computer timestamp" in df.columns:
        return "Computer timestamp"
    # last resort: first numeric
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No suitable time column found. Provide --time_col.")

def convert_to_seconds(series: pd.Series) -> pd.Series:
    """Detect units by magnitude and convert to seconds, then normalize to start at 0."""
    ts = pd.to_numeric(series, errors="coerce")
    maxv = np.nanmax(ts)
    if maxv > 1e12:
        scale = 1e9   # nanoseconds -> s
    elif maxv > 1e9:
        scale = 1e6   # microseconds -> s
    elif maxv > 1e7:
        scale = 1e3   # milliseconds -> s
    else:
        scale = 1.0   # seconds
    ts_s = ts / scale
    # normalize
    ts_s = ts_s - np.nanmin(ts_s)
    return ts_s

def duration_to_seconds(series: pd.Series) -> pd.Series:
    """Convert Tobii event duration to seconds by magnitude."""
    dur = pd.to_numeric(series, errors="coerce")
    maxv = np.nanmax(dur)
    if np.isnan(maxv):
        return pd.Series([np.nan]*len(dur), index=dur.index)
    if maxv > 1e12:
        scale = 1e9
    elif maxv > 1e9:
        scale = 1e6
    elif maxv > 1e7:
        scale = 1e3
    else:
        scale = 1.0
    return dur / scale

def window_count_saccades(sacc_df, ts_col, seg_df, offset):
    """Return total saccade rows inside all windows with given offset."""
    t = sacc_df[ts_col] + offset
    total = 0
    for _, seg in seg_df.iterrows():
        mask = (t >= seg["start_s"]) & (t < seg["end_s"])
        total += int(mask.sum())
    return total

def auto_align_offset(sacc_df, ts_col, seg_df, search_secs=300, step=0.5, verbose=False):
    """Search offset in [-search_secs, +search_secs] that maximizes captured saccade rows."""
    best_off = 0.0
    best_count = -1
    # Coarse then refine for speed (optional)
    # coarse step
    offs = np.arange(-search_secs, search_secs + 1e-9, step)
    for off in offs:
        c = window_count_saccades(sacc_df, ts_col, seg_df, off)
        if c > best_count:
            best_count = c
            best_off = off
    if verbose:
        print(f"[auto-align] best_offset={best_off:+.2f}s (max sacc rows inside windows={best_count})")
    return best_off, best_count

def compute_per_segment(sacc_df, ts_col, idx_col, dur_s_col, seg_df, offset, verbose=False):
    rows = []
    for _, seg in seg_df.iterrows():
        start, end = seg["start_s"], seg["end_s"]
        dur = end - start if end > start else np.nan
        mask = ((sacc_df[ts_col] + offset) >= start) & ((sacc_df[ts_col] + offset) < end)
        sub = sacc_df.loc[mask].copy()

        # unique events by index if present
        if idx_col in sub.columns and sub[idx_col].notna().any():
            # count unique indices
            uniq = sub.dropna(subset=[idx_col]).copy()
            uniq[idx_col] = pd.to_numeric(uniq[idx_col], errors="coerce")
            uniq = uniq.dropna(subset=[idx_col])
            event_ids = uniq[idx_col].unique()
            sacc_count = len(event_ids)

            # duration per event: take first non-null duration for each index
            sacc_total_dur = np.nan
            sacc_mean_dur = np.nan
            if dur_s_col in sub.columns:
                # per event
                dsub = sub.dropna(subset=[idx_col]).copy()
                dsub[idx_col] = pd.to_numeric(dsub[idx_col], errors="coerce")
                dsub = dsub.dropna(subset=[idx_col])
                if dsub[dur_s_col].notna().any():
                    # pick first value per event id
                    d_first = (
                        dsub.dropna(subset=[dur_s_col])
                            .sort_values(by=[idx_col, ts_col])
                            .groupby(idx_col, as_index=False)[dur_s_col].first()
                    )
                    sacc_total_dur = float(np.nansum(d_first[dur_s_col].values)) if len(d_first) else 0.0
                    sacc_mean_dur  = float(np.nanmean(d_first[dur_s_col].values)) if len(d_first) else np.nan
        else:
            # fallback: approximate events by transitions (contiguous runs)
            # This is less reliable; better to rely on the index.
            sub = sub.sort_values(by=ts_col)
            sacc_count = 0
            last_seen = False
            for _i in range(len(sub)):
                # every row is saccade here
                if not last_seen:
                    sacc_count += 1
                last_seen = True
            sacc_total_dur = np.nan
            sacc_mean_dur  = np.nan

        sacc_per_sec = float(sacc_count / dur) if dur and dur > 0 else 0.0

        if verbose:
            print(f"seg {seg['segment_id']}: rows={len(sub):4d}, events={sacc_count:4d}, dur_win={dur:.2f}s, sacc_total_dur={sacc_total_dur if not np.isnan(sacc_total_dur) else 0:.3f}s")

        rows.append({
            "segment_id": int(seg["segment_id"]),
            "duration_s": float(dur) if dur else np.nan,
            "saccade_count": int(sacc_count),
            "saccades_per_sec": sacc_per_sec,
            "sacc_total_dur_s": None if (pd.isna(sacc_total_dur)) else float(sacc_total_dur),
            "sacc_mean_dur_s": None if (pd.isna(sacc_mean_dur)) else float(sacc_mean_dur),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Extract saccade metrics per segment with auto-alignment.")
    ap.add_argument("--participant_id", required=True, help="e.g. P03")
    ap.add_argument("--base_dir", default=DEFAULT_BASE)
    ap.add_argument("--tsv", default=None, help="Optional direct path to TSV")
    ap.add_argument("--time_col", default=None, help='e.g., "Recording timestamp" or "Computer timestamp"')
    ap.add_argument("--search_secs", type=float, default=300.0, help="Offset search range in seconds (+/-)")
    ap.add_argument("--step", type=float, default=0.5, help="Offset search step (seconds)")
    ap.add_argument("--no_autoalign", action="store_true", help="Skip search; use --offset_seconds as-is")
    ap.add_argument("--offset_seconds", type=float, default=0.0, help="Manual offset to add to TSV time before windowing")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    base = args.base_dir
    pid  = args.participant_id

    # 1) YAML segments (need start/end)
    seg_df = find_segments_yaml(base, pid)

    # 2) TSV
    tsv_path = find_tsv(base, pid, override=args.tsv)
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    # 3) pick time col & convert to seconds (normalized)
    tcol = choose_time_col(df, preferred=args.time_col)
    df["_time_s"] = convert_to_seconds(df[tcol])

    # 4) filter saccade rows
    if TYPE_COL not in df.columns:
        raise ValueError(f"Column '{TYPE_COL}' not found in TSV.")
    sacc = df.loc[df[TYPE_COL] == SACCADE_LABEL].copy()

    # normalize duration column if available
    dur_s_col = None
    if DUR_COL in sacc.columns:
        sacc["_dur_s"] = duration_to_seconds(sacc[DUR_COL])
        dur_s_col = "_dur_s"

    # 5) auto-align or manual offset
    if args.no_autoalign:
        best_off = float(args.offset_seconds)
        if args.debug:
            print(f"[manual offset] using offset={best_off:+.2f}s")
    else:
        best_off, best_count = auto_align_offset(
            sacc_df=sacc, ts_col="_time_s", seg_df=seg_df,
            search_secs=args.search_secs, step=args.step, verbose=args.debug
        )
        # Optionally allow a manual tweak on top
        best_off += float(args.offset_seconds)

    # 6) compute per-segment metrics
    out_df = compute_per_segment(
        sacc_df=sacc,
        ts_col="_time_s",
        idx_col=IDX_COL if IDX_COL in sacc.columns else None,
        dur_s_col=dur_s_col,
        seg_df=seg_df,
        offset=best_off,
        verbose=args.debug
    )

    # 7) Save
    out_dir = os.path.join(base, pid, "objective_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{pid}_saccades_by_segment.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}")

if __name__ == "__main__":
    main()
