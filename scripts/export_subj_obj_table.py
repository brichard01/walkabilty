#!/usr/bin/env python
import argparse, os, re, ast
import pandas as pd
import dataframe_image as dfi

def clean_np_float64_list(val):
    """Strip out np.float64(...) wrappers before literal_eval."""
    if isinstance(val, str):
        cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', val)
        return ast.literal_eval(cleaned)
    return val

def main():
    p = argparse.ArgumentParser(
        description="Export Subjective+Objective correlation table as PNG"
    )
    p.add_argument(
        "--participant_id", "-p",
        required=True,
        help="Participant ID, e.g. P01, P02, …"
    )
    args = p.parse_args()
    pid = args.participant_id

    # ── Paths ─────────────────────────────────────────────
    base_dir = r"C:/Abderrahim Internship Data/Data"
    in_csv   = os.path.join(base_dir, pid, f"{pid}_final_analysis.csv")
    out_png  = os.path.join(base_dir, pid, "plots", f"{pid}_subj_obj_table.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Could not find final analysis CSV: {in_csv}")

    # ── Load & preprocess ──────────────────────────────────
    df = pd.read_csv(in_csv)

    # parse the list-columns
    df['top_classes']     = df['top_classes'].apply(ast.literal_eval)
    df['top_percentages'] = df['top_percentages'].apply(clean_np_float64_list)

    # build a single “Top Gaze Fixation” column
    gaze_strs = []
    for labs, vals in zip(df['top_classes'], df['top_percentages']):
        parts = [f"{lab} ({v:.1f}%)" for lab, v in zip(labs, vals)]
        gaze_strs.append(" • ".join(parts))
    df['Top Gaze Fixation'] = gaze_strs

    # select & rename columns for display
    table = df[[
        'segment_id',
        'walkability_rating',
        'feedback',
        'themes',
        'Top Gaze Fixation'
    ]].copy()
    table.columns = [
        'Segment',
        'Walkability\nRating',
        'Feedback',
        'Themes',
        'Top Gaze Fixation'
    ]

    # ── Style & export ─────────────────────────────────────
    styler = (
        table.style
             .set_table_styles([
                 {'selector':'th', 'props':[
                     ('font-size','12pt'),
                     ('text-align','center'),
                     ('border','1px solid black')
                 ]},
                 {'selector':'td', 'props':[
                     ('vertical-align','top'),
                     ('text-align','left'),
                     ('font-size','10pt'),
                     ('border','1px solid #ddd')
                 ]}
             ])
             .hide(axis="index")    # hides the row index
             .set_properties(subset=['Segment','Walkability\nRating'],
                             **{'width':'60px','text-align':'center'})
             .set_properties(subset=['Feedback','Themes','Top Gaze Fixation'],
                             **{'width':'300px'})
    )

    dfi.export(
        styler,
        out_png,
        table_conversion="matplotlib",
        dpi=300
    )

    print(f"✅ Saved combined table to {out_png}")

if __name__ == "__main__":
    main()
