#!/usr/bin/env python
import os, ast, argparse, textwrap, math, re
import pandas as pd
import matplotlib.pyplot as plt

def clean_percentage_list(cell):
    if not isinstance(cell, str):
        return cell
    # strip np.float64(...) wrappers
    s = re.sub(r'np\.float64\((.*?)\)', r'\1', cell)
    return ast.literal_eval(s)

def load_final(base_dir, pid):
    fn = os.path.join(base_dir, pid, f"{pid}_final_analysis.csv")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Missing final_analysis.csv: {fn}")
    df = pd.read_csv(fn, dtype={"segment_id":int})
    df["top_classes"]     = df["top_classes"].apply(ast.literal_eval)
    df["top_percentages"] = df["top_percentages"].apply(clean_percentage_list)
    df["themes"]          = df["themes"].apply(ast.literal_eval)
    return df.set_index("segment_id")

def make_segment_cards(df, pid, out_png):
    segs = sorted(df.index)
    n = len(segs)
    cols = 2
    rows = math.ceil(n/cols)

    fig = plt.figure(figsize=(10*cols, 5*rows))
    gs = fig.add_gridspec(rows, cols, wspace=0.8, hspace=0.8)

    for i, seg in enumerate(segs):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        rec = df.loc[seg]

        # Top 3 fixations
        top3 = rec["top_classes"][:3]
        pct3 = rec["top_percentages"][:3]
        ys = list(range(len(top3)))[::-1]
        ax.barh(ys, pct3, color="#7FB3D5")
        ax.set_yticks(ys)
        ax.set_yticklabels(top3, fontsize=10)
        ax.set_xlim(0, max(pct3)*1.2)
        ax.set_xlabel("Fixation %", fontsize=9)

        # Title
        ax.set_title(f"Segment {seg}  —  Rating {rec['walkability_rating']:.1f}",
                     fontsize=12, fontweight="bold")

        # Build text
        theme_lines = "\n".join(f"• {t}" for t in rec["themes"])
        fb = textwrap.fill(rec["feedback"].strip(), width=40)
        txt = f"Themes:\n{theme_lines}\n\n“{fb}”"

        # Place text as an inset with axis coords
        ax.text(1.01, 0.98, txt,
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#FDFEFE", edgecolor="#D5DBDB"))

        # tidy up
        for sp in ["top","right"]:
            ax.spines[sp].set_visible(False)

    fig.suptitle(f"{pid} — Segment Cards", fontsize=18, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved segment cards to {out_png}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--participant_id", required=True)
    p.add_argument("--base_dir", default=r"C:/Abderrahim Internship Data/Data")
    args = p.parse_args()

    df_final = load_final(args.base_dir, args.participant_id)
    out_png = os.path.join(args.base_dir,
                           args.participant_id,
                           "plots",
                           f"{args.participant_id}_segment_cards.png")
    make_segment_cards(df_final, args.participant_id, out_png)

if __name__=="__main__":
    main()
