#!/usr/bin/env python
import pandas as pd
import os
import ast
import argparse

def parse_theme_column(cell):
    try:
        return ast.literal_eval(cell)
    except:
        return [s.strip() for s in cell.split(",")]

def main():
    # ─── CLI ─────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Merge subjective and objective segment-level data into a final analysis table."
    )
    parser.add_argument(
        "--participant_id",
        required=True,
        help="Participant ID, e.g. P01, P02, etc."
    )
    args = parser.parse_args()
    pid = args.participant_id

    # ─── PATHS ────────────────────────────────────────────────────────────
    base_dir = r"C:/Abderrahim Internship Data/Data"
    subj_csv = os.path.join(base_dir, pid, "subjective", f"{pid}_subjective.csv")
    gaze_csv = os.path.join(base_dir, pid, "objective_analysis", f"{pid}_segment_fixation_percentage.csv")
    output_csv = os.path.join(base_dir, pid, f"{pid}_final_analysis.csv")

    # ─── LOAD ─────────────────────────────────────────────────────────────
    if not os.path.exists(subj_csv):
        raise FileNotFoundError(f"Subjective file not found: {subj_csv}")
    if not os.path.exists(gaze_csv):
        raise FileNotFoundError(f"Gaze percentages file not found: {gaze_csv}")

    df_subjective = pd.read_csv(subj_csv)
    df_gaze      = pd.read_csv(gaze_csv)

    # ─── PREP ─────────────────────────────────────────────────────────────
    # ensure segment_id is integer
    df_subjective["segment_id"] = df_subjective["segment_id"].astype(int)
    df_gaze["segment_id"]       = df_gaze["segment_id"].astype(int)
    df_gaze.set_index("segment_id", inplace=True)

    # parse themes
    df_subjective['themes'] = df_subjective['themes'].apply(parse_theme_column)

    # ─── TOP‐N EXTRACT ─────────────────────────────────────────────────────
    top_n = 5
    rows = []
    for seg in df_gaze.index:
        row = df_gaze.loc[seg]
        top = row.sort_values(ascending=False).head(top_n)
        rows.append({
            "segment_id": seg,
            "top_classes":    list(top.index),
            "top_percentages": list(top.values)
        })
    df_top = pd.DataFrame(rows)

    # ─── MERGE & SAVE ────────────────────────────────────────────────────
    df_final = pd.merge(df_subjective, df_top, on="segment_id")
    df_final.to_csv(output_csv, index=False)
    print(f"✅ Saved final analysis table to {output_csv}")

if __name__ == "__main__":
    main()
