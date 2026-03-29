#!/usr/bin/env python
import os, argparse, ast, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def clean_np_float64_list(val):
    if isinstance(val, str):
        cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', val)
        return ast.literal_eval(cleaned)
    return val

def main():
    p = argparse.ArgumentParser(
        description="Combined objective/subjective summary per segment"
    )
    p.add_argument("--participant_id", required=True)
    args = p.parse_args()
    pid = args.participant_id

    base = r"C:/Abderrahim Internship Data/Data"
    fin = os.path.join(base, pid, f"{pid}_final_analysis.csv")
    out = os.path.join(base, pid, "plots", f"{pid}_summary.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if not os.path.exists(fin):
        raise FileNotFoundError(fin)
    df = pd.read_csv(fin)

    # parse list columns
    df['top_classes']     = df['top_classes'].apply(ast.literal_eval)
    df['top_percentages'] = df['top_percentages'].apply(clean_np_float64_list)
    df['themes']          = df['themes'].apply(lambda cell: ast.literal_eval(cell) 
                                               if isinstance(cell, str) and cell.strip().startswith('[') 
                                               else [s.strip() for s in str(cell).split(',')])

    # build a stacked DataFrame of only the top N classes
    top_n = 5
    segs = []
    for _,r in df.iterrows():
        d = dict(zip(r['top_classes'][:top_n], r['top_percentages'][:top_n]))
        d['segment_id'] = r['segment_id']
        segs.append(d)
    pct = pd.DataFrame(segs).fillna(0).set_index('segment_id')

    # prepare layout
    plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[3, 0.3, 1], hspace=0.3)

    # ── Top: Stacked bar chart ─────────────────────────
    ax = plt.subplot(gs[0])
    pct.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', width=0.8)
    # overlay line for walkability
    walk = df.set_index('segment_id')['walkability_rating']
    ax2  = ax.twinx()
    walk.plot(ax=ax2, color='black', marker='o', linewidth=2, label='Walkability')
    ax.set_ylabel("Fixation %", fontsize=12)
    ax2.set_ylabel("Walkability", fontsize=12)
    ax.set_title(f"{pid} — Top {top_n} Gaze Classes vs. Walkability", fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 0.9))

    # ── Middle: blank spacer ───────────────────────────
    plt.subplot(gs[1]).axis('off')

    # ── Bottom: key feedback + themes table ────────────
    ax3 = plt.subplot(gs[2])
    ax3.axis('off')

    # build table rows
    table_data = []
    for _, r in df.iterrows():
        sid = r['segment_id']
        fb  = r['feedback'].strip().replace('\n',' ')[:80] + "…"
        ths = "; ".join(r['themes'])
        table_data.append([sid, fb, ths])
    col_labels = ["Segment", "Feedback (→ first 80 chars)", "Themes"]
    table = ax3.table(cellText=table_data,
                      colLabels=col_labels,
                      cellLoc='left',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅ Saved summary to {out}")

if __name__ == "__main__":
    main()
