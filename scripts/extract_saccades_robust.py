#!/usr/bin/env python
import os, argparse, yaml
import pandas as pd
import numpy as np

DEFAULT_BASE = r"C:/Abderrahim Internship Data/Data"

def hms_to_s(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h)*3600 + int(m)*60 + float(s)

def load_segments(base_dir, pid):
    yml = os.path.join(base_dir, pid, f"segments_{pid}.yaml")
    with open(yml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rows = []
    for seg in data.get("segments", []):
        rows.append({
            "segment_id": int(seg["segment_id"]),
            "start_s": hms_to_s(seg["start_time"]),
            "end_s":   hms_to_s(seg["end_time"]),
        })
    seg_df = pd.DataFrame(rows).sort_values("segment_id")
    seg_df["duration_s"] = seg_df["end_s"] - seg_df["start_s"]
    return seg_df

def load_tsv(base_dir, pid, file_name=None):
    if file_name is None:
        file_name = f"{pid}.tsv"
    path = os.path.join(base_dir, pid, "raw", file_name)
    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False, engine="python", on_bad_lines="skip")
    return df

def robust_bool_saccade(df, type_col, index_col, sacc_idx):
    sacc_text = pd.Series(False, index=df.index)
    if type_col in df.columns:
        s = df[type_col].astype(str).str.lower()
        sacc_text = s.str.contains("sacc")
    sacc_idx_match = pd.Series(False, index=df.index)
    if index_col and index_col in df.columns:
        ix = pd.to_numeric(df[index_col], errors="coerce")
        sacc_idx_match = ix.eq(sacc_idx)
    return (sacc_text | sacc_idx_match)

def summarize_saccades_for_segment(df_seg, ts_col_ms, dur_ms_col, sacc_mask):
    if df_seg.empty:
        return 0, 0.0, np.nan
    ts = pd.to_numeric(df_seg[ts_col_ms], errors="coerce")/1000.0
    m = sacc_mask.loc[df_seg.index].fillna(False).to_numpy()

    prev = np.concatenate(([False], m[:-1]))
    starts = (~prev & m)
    n_events = int(starts.sum())

    total_dur = 0.0
    mean_dur = np.nan
    if dur_ms_col and dur_ms_col in df_seg.columns:
        dur_ms = pd.to_numeric(df_seg[dur_ms_col], errors="coerce").fillna(0.0).to_numpy()
        ends = (m & np.concatenate((m[1:] == False, [True])))
        total_dur = float((dur_ms * ends).sum() / 1000.0)
        if n_events > 0:
            mean_dur = total_dur / n_events
    else:
        dt = np.diff(ts, prepend=ts.iloc[0])
        total_dur = float((dt * m).sum())
        if n_events > 0:
            mean_dur = total_dur / n_events

    return n_events, total_dur, mean_dur

def main():
    ap = argparse.ArgumentParser(description="Robust saccade counts/durations per segment from Tobii TSV.")
    ap.add_argument("--participant_id", required=True)
    ap.add_argument("--base_dir", default=DEFAULT_BASE)
    ap.add_argument("--time_col", default="Recording timestamp")
    ap.add_argument("--type_col", default="Eye movement type")
    ap.add_argument("--index_col", default="Eye movement type index")
    ap.add_argument("--dur_col", default="Eye movement event duration")
    ap.add_argument("--sacc_idx", type=int, default=2)
    ap.add_argument("--normalize_time", action="store_true",
                    help="Shift the chosen time column so the first sample is t=0s (fixes offset mismatches).")
    ap.add_argument("--debug", action="store_true",
                    help="Print per-segment sample counts and saccade frames.")
    ap.add_argument("--print_types", action="store_true",
                    help="Show value counts of Eye movement type and exit.")
    args = ap.parse_args()

    pid = args.participant_id
    base = args.base_dir

    seg_df = load_segments(base, pid)
    tsv = load_tsv(base, pid)

    if args.print_types and args.type_col in tsv.columns:
        print("\nEye movement type (top):")
        print(pd.Series(tsv[args.type_col]).value_counts(dropna=False).head(40))
        return

    # pick time col & make seconds
    if args.time_col not in tsv.columns:
        raise ValueError(f"Time column not found: {args.time_col}")
    ts_s = pd.to_numeric(tsv[args.time_col], errors="coerce") / 1000.0
    if args.normalize_time:
        ts_s = ts_s - np.nanmin(ts_s)
    tsv["_ts_s"] = ts_s

    sacc_mask = robust_bool_saccade(tsv, args.type_col, args.index_col, args.sacc_idx)

    out_rows = []
    for r in seg_df.itertuples(index=False):
        seg_mask = (tsv["_ts_s"] >= r.start_s) & (tsv["_ts_s"] < r.end_s)
        df_slice = tsv.loc[seg_mask]
        n, total_s, mean_s = summarize_saccades_for_segment(
            df_slice, args.time_col, args.dur_col, sacc_mask
        )
        duration_s = float(r.duration_s)
        rate = (n / duration_s) if duration_s > 0 else np.nan
        if args.debug:
            # show how many samples and how many saccade frames are inside
            n_samples = len(df_slice)
            n_sacc_frames = int(sacc_mask.loc[df_slice.index].sum()) if n_samples else 0
            print(f"seg {r.segment_id}: samples={n_samples}, sacc_frames={n_sacc_frames}, events={n}, total_dur={total_s:.3f}s")

        out_rows.append({
            "segment_id": r.segment_id,
            "duration_s": duration_s,
            "saccade_count": n,
            "saccades_per_sec": rate,
            "sacc_total_dur_s": total_s,
            "sacc_mean_dur_s": mean_s,
        })

    out = pd.DataFrame(out_rows)
    out_path = os.path.join(base, pid, "objective_analysis", f"{pid}_saccades_by_segment.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
