#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract saccade metrics per segment using linear time alignment:
  t_segment ≈ a + b * t_tsv

Outputs:
  <BASE>/<PID>/objective_analysis/<PID>_saccades_by_segment.csv

Run:
  python extract_saccades_linear_align.py --participant_id P07 --base_dir "C:/Abderrahim Internship Data/Data" --debug
  # (You can try the other time column if needed)
  python extract_saccades_linear_align.py --participant_id P07 --base_dir "C:/Abderrahim Internship Data/Data" --time_col "Computer timestamp" --debug
"""

import os, glob, argparse
import numpy as np
import pandas as pd
import yaml

DEFAULT_BASE = r"C:/Abderrahim Internship Data/Data"

# Tobii column names
TYPE_COL = "Eye movement type"
IDX_COL  = "Eye movement type index"
DUR_COL  = "Eye movement event duration"
REC_DUR_COL = "Recording duration"

SACCADE = "Saccade"

# ----------------------- helpers -----------------------

def hhmmss_to_seconds(hhmmss: str) -> float:
    hh, mm, ss = hhmmss.strip().split(":")
    return int(hh)*3600 + int(mm)*60 + float(ss)

def load_segments_yaml(base_dir, pid):
    yml = os.path.join(base_dir, pid, f"segments_{pid}.yaml")
    if not os.path.exists(yml):
        raise FileNotFoundError(f"Segments YAML not found: {yml}")
    with open(yml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rows = []
    for seg in data.get("segments", []):
        rows.append({
            "segment_id": int(seg["segment_id"]),
            "start_s": hhmmss_to_seconds(seg["start_time"]),
            "end_s":   hhmmss_to_seconds(seg["end_time"]),
        })
    seg_df = pd.DataFrame(rows).sort_values("segment_id")
    seg_df["dur_s"] = seg_df["end_s"] - seg_df["start_s"]
    return seg_df

def find_tsv(base_dir, pid, override=None):
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"TSV override not found: {override}")
        return override
    raw_dir = os.path.join(base_dir, pid, "raw")
    cands = glob.glob(os.path.join(raw_dir, "*.tsv"))
    if not cands:
        cands = glob.glob(os.path.join(base_dir, pid, "*.tsv"))
    if not cands:
        raise FileNotFoundError(f"No TSV found in {raw_dir} or {os.path.join(base_dir,pid)}")
    return cands[0]

def choose_time_col(df, preferred=None):
    if preferred and preferred in df.columns:
        return preferred
    if "Recording timestamp" in df.columns:
        return "Recording timestamp"
    if "Computer timestamp" in df.columns:
        return "Computer timestamp"
    # fallback: first numeric
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No suitable time column found. Provide --time_col.")

def to_seconds(series: pd.Series) -> pd.Series:
    """Detect units (ns/us/ms/s) and normalize to start at 0s."""
    x = pd.to_numeric(series, errors="coerce")
    maxv = np.nanmax(x)
    if maxv > 1e12:
        scale = 1e9   # ns -> s
    elif maxv > 1e9:
        scale = 1e6   # us -> s
    elif maxv > 1e7:
        scale = 1e3   # ms -> s
    else:
        scale = 1.0   # s
    xs = x / scale
    return xs - np.nanmin(xs)

def dur_to_seconds(series: pd.Series) -> pd.Series:
    d = pd.to_numeric(series, errors="coerce")
    maxv = np.nanmax(d)
    if np.isnan(maxv):
        return pd.Series([np.nan]*len(d), index=series.index)
    if maxv > 1e12:
        scale = 1e9
    elif maxv > 1e9:
        scale = 1e6
    elif maxv > 1e7:
        scale = 1e3
    else:
        scale = 1.0
    return d / scale

def tsv_duration_seconds(df, tcol):
    """Prefer explicit 'Recording duration' if present; else range of tcol seconds."""
    if REC_DUR_COL in df.columns:
        d = pd.to_numeric(df[REC_DUR_COL], errors="coerce")
        if d.notna().any():
            # detect units and return single value (max)
            return float(np.nanmax(dur_to_seconds(d)))
    # fallback to time column span
    ts = to_seconds(df[tcol])
    return float(np.nanmax(ts) - np.nanmin(ts))

def build_saccade_events(df_sacc, time_col_s, debug=False):
    """
    Build event-level table using Eye movement type index (preferred). Each row = one saccade event.
    time_s = first timestamp for that event; dur_s = first non-null duration value for that event.
    If index missing, fallback: treat each row as a tiny event (less ideal).
    """
    if IDX_COL in df_sacc.columns and df_sacc[IDX_COL].notna().any():
        tmp = df_sacc.copy()
        tmp[IDX_COL] = pd.to_numeric(tmp[IDX_COL], errors="coerce")
        tmp = tmp.dropna(subset=[IDX_COL])
        # sort by time to make "first" meaningful
        tmp = tmp.sort_values(time_col_s)
        # first time per event
        first_time = tmp.groupby(IDX_COL, as_index=False)[time_col_s].first()
        # duration: take first non-null per event if available
        if DUR_COL in tmp.columns:
            tmp["_dur_s"] = dur_to_seconds(tmp[DUR_COL])
            d_first = tmp.dropna(subset=["_dur_s"]).groupby(IDX_COL, as_index=False)["_dur_s"].first()
            ev = pd.merge(first_time, d_first, on=IDX_COL, how="left")
            ev.rename(columns={time_col_s: "time_s", "_dur_s": "dur_s"}, inplace=True)
        else:
            ev = first_time.rename(columns={time_col_s: "time_s"})
            ev["dur_s"] = np.nan
        if debug:
            print(f"[events] unique saccade events from index: {len(ev)}")
        return ev[[IDX_COL, "time_s", "dur_s"]].copy()
    else:
        # Fallback: 1 row = 1 event (crude)
        tmp = df_sacc.sort_values(time_col_s).copy()
        tmp.rename(columns={time_col_s: "time_s"}, inplace=True)
        tmp["dur_s"] = dur_to_seconds(tmp[DUR_COL]) if (DUR_COL in tmp.columns) else np.nan
        if debug:
            print(f"[events] no index found -> using row-level events: {len(tmp)}")
        # fabricate an index
        tmp[IDX_COL] = np.arange(len(tmp))
        return tmp[[IDX_COL, "time_s", "dur_s"]].copy()

def score_alignment(ev, seg_df, a, b):
    """Count how many events fall inside segments after mapping t' = a + b*t."""
    tt = a + b * ev["time_s"].values
    count = 0
    for _, s in seg_df.iterrows():
        mask = (tt >= s["start_s"]) & (tt < s["end_s"])
        count += int(mask.sum())
    return count

def best_linear_alignment(ev, seg_df, tsv_dur_s, search_offset=120.0, step_offset=0.25,
                          base_scale=None, scale_jitter=0.05, scale_step=0.01, debug=False):
    """
    Find (a, b) maximizing events-in-windows.
    b starts from base_scale (segments_total / tsv_dur), then we try b in [b*(1-δ) .. b*(1+δ)].
    a is searched in [-search_offset .. +search_offset] with step_offset.
    """
    seg_total = float(seg_df["dur_s"].sum())
    if base_scale is None or not np.isfinite(base_scale) or base_scale <= 0:
        base_scale = seg_total / max(tsv_dur_s, 1e-6)

    b_vals = np.arange(base_scale * (1 - scale_jitter),
                       base_scale * (1 + scale_jitter) + 1e-9,
                       scale_step)
    a_vals = np.arange(-search_offset, search_offset + 1e-9, step_offset)

    best = {"a": 0.0, "b": base_scale, "hits": -1}
    for b in b_vals:
        for a in a_vals:
            h = score_alignment(ev, seg_df, a, b)
            if h > best["hits"]:
                best = {"a": float(a), "b": float(b), "hits": int(h)}
    if debug:
        ratio = best["hits"] / max(len(ev), 1)
        print(f"[align] best a={best['a']:+.2f}s, b={best['b']:.4f}, hits={best['hits']}/{len(ev)} ({ratio:.1%})")
    return best["a"], best["b"]

def assign_and_aggregate(ev, seg_df, a, b, debug=False):
    """Assign events to segments via t' = a + b*t, aggregate per segment."""
    ev = ev.copy()
    ev["t_map"] = a + b * ev["time_s"].values

    rows = []
    for _, s in seg_df.iterrows():
        mask = (ev["t_map"] >= s["start_s"]) & (ev["t_map"] < s["end_s"])
        sub = ev.loc[mask]

        count = int(len(sub))
        dur_s = s["dur_s"] if s["dur_s"] > 0 else np.nan
        per_sec = (count / dur_s) if (dur_s and np.isfinite(dur_s) and dur_s > 0) else 0.0

        # duration stats
        if sub["dur_s"].notna().any():
            total_dur = float(np.nansum(sub["dur_s"].values))
            mean_dur  = float(np.nanmean(sub["dur_s"].values))
        else:
            total_dur = None
            mean_dur  = None

        rows.append({
            "segment_id": int(s["segment_id"]),
            "duration_s": float(s["dur_s"]),
            "saccade_count": count,
            "saccades_per_sec": per_sec,
            "sacc_total_dur_s": total_dur,
            "sacc_mean_dur_s": mean_dur,
        })

        if debug:
            print(f"seg {s['segment_id']}: events={count:4d} | window={s['dur_s']:.1f}s | per_sec={per_sec:.3f}")

    return pd.DataFrame(rows)

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Saccades per segment using linear time alignment (a + b*t).")
    ap.add_argument("--participant_id", required=True, help="e.g., P07")
    ap.add_argument("--base_dir", default=DEFAULT_BASE)
    ap.add_argument("--tsv", default=None, help="Optional direct path to TSV")
    ap.add_argument("--time_col", default=None, help='Force time column, e.g. "Computer timestamp"')
    ap.add_argument("--search_offset", type=float, default=120.0, help="Offset search half-range in seconds")
    ap.add_argument("--offset_step", type=float, default=0.25, help="Offset search step")
    ap.add_argument("--scale_jitter", type=float, default=0.05, help="± percent around base scale (e.g., 0.05=±5%)")
    ap.add_argument("--scale_step", type=float, default=0.01, help="Step for scale search")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    base = args.base_dir
    pid  = args.participant_id

    # 1) Segments
    seg_df = load_segments_yaml(base, pid)
    seg_total = float(seg_df["dur_s"].sum())

    # 2) TSV
    tsv_path = find_tsv(base, pid, args.tsv)
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    # 3) time column -> seconds
    tcol = choose_time_col(df, args.time_col)
    df["_time_s_raw"] = to_seconds(df[tcol])
    tsv_dur_s = tsv_duration_seconds(df, tcol)

    # 4) filter Saccade rows
    if TYPE_COL not in df.columns:
        raise ValueError(f"Column '{TYPE_COL}' missing in TSV.")
    sacc = df.loc[df[TYPE_COL] == SACCADE].copy()

    # 5) Build events
    ev = build_saccade_events(sacc, time_col_s="_time_s_raw", debug=args.debug)

    # 6) Alignment (a,b)
    base_scale = seg_total / max(tsv_dur_s, 1e-6)
    a, b = best_linear_alignment(
        ev, seg_df, tsv_dur_s,
        search_offset=args.search_offset,
        step_offset=args.offset_step,
        base_scale=base_scale,
        scale_jitter=args.scale_jitter,
        scale_step=args.scale_step,
        debug=args.debug
    )

    # 7) Aggregate per segment
    out = assign_and_aggregate(ev, seg_df, a, b, debug=args.debug)

    # 8) Save
    out_dir = os.path.join(base, pid, "objective_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{pid}_saccades_by_segment.csv")
    out.to_csv(out_csv, index=False)

    if args.debug:
        total_events = len(ev)
        captured = int(out["saccade_count"].sum())
        print(f"[summary] total events={total_events}, captured in segments={captured} ({captured/max(total_events,1):.1%})")
        print(f"[mapping] t_segment ≈ {a:+.2f} + {b:.4f} * t_tsv   (tsv_dur={tsv_dur_s:.1f}s, seg_total={seg_total:.1f}s)")

    print(f"✅ Saved: {out_csv}")

if __name__ == "__main__":
    main()
