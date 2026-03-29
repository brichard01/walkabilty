#!/usr/bin/env python
import argparse, os, sys, textwrap
import pandas as pd

def try_import_dfi():
    try:
        import dataframe_image as dfi
        return dfi
    except Exception:
        return None

def wrap_col(series, width):
    return series.apply(lambda s: "\n".join(textwrap.wrap(str(s), width=width)))

def main():
    ap = argparse.ArgumentParser(description="Render *_best_worst_summary.csv to a PNG table")
    ap.add_argument("--participant_id", required=True, help="e.g., P01")
    ap.add_argument("--base_dir", required=True, help='e.g., "C:/Abderrahim Internship Data/Data"')
    ap.add_argument("--wrap", type=int, default=50, help="wrap width for text columns")
    args = ap.parse_args()

    pid = args.participant_id
    csv_path = os.path.join(args.base_dir, pid, "plots", f"{pid}_best_worst_summary.csv")
    out_png  = os.path.join(args.base_dir, pid, "plots", f"{pid}_best_worst_summary_table.png")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found:\n  {csv_path}")

    df = pd.read_csv(csv_path)

    # Nicely wrapped columns
    for c in ["themes", "feedback", "top_classes"]:
        if c in df.columns:
            df[c] = wrap_col(df[c], args.wrap)

    # Reorder/rename for clarity
    keep_cols = ["which", "segment_id", "rating", "themes", "feedback", "top_classes"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df = df.rename(columns={
        "which": "Segment Type",
        "segment_id": "Segment",
        "rating": "Walkability",
        "themes": "Key themes",
        "feedback": "Key sentence(s)",
        "top_classes": "Top objective classes (Fixation %)"
    })

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    dfi = try_import_dfi()
    if dfi:
        # dataframe_image path (best quality)
        styler = (df.style
                    .hide(axis="index")
                    .set_caption(f"{pid} — Best vs Worst Segment (subjective + objective)")
                    .set_properties(**{"white-space": "pre-wrap", "font-size": "11px"})
                  )
        dfi.export(styler, out_png, table_conversion="matplotlib")
        print(f"✅ Saved table PNG to {out_png}")
        return

    # Fallback to matplotlib table if dataframe_image isn't installed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis("off")
    the_table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="upper left"
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 2.0)

    plt.title(f"{pid} — Best vs Worst Segment (subjective + objective)", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"✅ Saved table PNG to {out_png}")

if __name__ == "__main__":
    main()
