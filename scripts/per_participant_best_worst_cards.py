#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clean 'segment cards' (subjective + objective) for the BEST and WORST segments.
- Text sits ABOVE the bars (no overlap).
- Long text is wrapped & clipped.
- Works with either {pid}/subjective/{pid}_subjective.csv OR segments_{pid}.yaml
- Objective classes from {pid}/objective_analysis/{pid}_segment_fixation_percentage.csv

Usage example:
  python per_participant_best_worst_cards.py --participant_id P01 \
    --base_dir "C:/Abderrahim Internship Data/Data" \
    --fig_w 15 --fig_h 8 --wrap 72 --text_lines 6
"""

import argparse, os, ast, yaml, textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_BASE = r"C:/Abderrahim Internship Data/Data"
IRRELEVANT   = {"Mountain","Sand","Snow","Boat"}  # drop if present


# ── helpers ──────────────────────────────────────────────────────────────────
def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]

def _wrap(text: str, width: int, max_lines: int) -> str:
    s = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if not s:
        return "—"
    lines = textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = (lines[-1] + "…") if not lines[-1].endswith("…") else lines[-1]
    return "\n".join(lines)


# ── loaders ─────────────────────────────────────────────────────────────────
def load_subjective(base: str, pid: str) -> pd.DataFrame:
    csv_path = os.path.join(base, pid, "subjective", f"{pid}_subjective.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = _drop_unnamed(df)
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

        df["themes"] = df["themes"].apply(parse_themes) if "themes" in df.columns else [[]]*len(df)
        if "feedback" not in df.columns: df["feedback"] = ""
        return df[["segment_id","walkability_rating","feedback","themes"]].copy()

    yml_path = os.path.join(base, pid, f"segments_{pid}.yaml")
    if not os.path.exists(yml_path):
        raise FileNotFoundError(f"Subjective not found:\n  {csv_path}\n  {yml_path}")
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

def load_objective_percent(base: str, pid: str) -> pd.DataFrame:
    f = os.path.join(base, pid, "objective_analysis", f"{pid}_segment_fixation_percentage.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Objective percentages CSV not found: {f}")
    df = pd.read_csv(f)
    df = _drop_unnamed(df)
    if "segment_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "segment_id"})
    df["segment_id"] = df["segment_id"].astype(int)
    df = df.set_index("segment_id")
    keep = [c for c in df.columns if c not in IRRELEVANT and not c.lower().startswith("unnamed")]
    df = df[keep].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


# ── logic ───────────────────────────────────────────────────────────────────
def pick_best_worst(subj: pd.DataFrame):
    best  = subj.sort_values(["walkability_rating","segment_id"], ascending=[False,True]).iloc[0]
    worst = subj.sort_values(["walkability_rating","segment_id"],  ascending=[True, True]).iloc[0]
    return best, worst


# ── plotting ────────────────────────────────────────────────────────────────
def _draw_card(fig, outer_cell, tag: str, meta: dict, top_series: pd.Series,
               wrap_chars: int, max_text_lines: int):
    """
    One side card: two rows (text block over bars). No overlap.
    """
    gs = outer_cell.subgridspec(nrows=2, ncols=1, height_ratios=[0.60, 0.40], hspace=0.30)
    ax_text = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    # TEXT
    ax_text.axis("off")
    title = f"{tag} – Segment {int(meta['segment_id'])} (Rating {float(meta['walkability_rating']):.1f})"
    ax_text.text(0.0, 1.00, title, fontsize=14, weight='bold', va='top', ha='left')

    themes = ", ".join(map(str, meta.get("themes", []))) or "—"
    themes_wrapped   = _wrap(themes, wrap_chars, 3)
    feedback_wrapped = _wrap(str(meta.get("feedback","")), wrap_chars, max_text_lines)

    ax_text.text(0.0, 0.80, "Themes:",   fontsize=12, weight="bold", va="top", ha="left")
    ax_text.text(0.10, 0.80, themes_wrapped, fontsize=11, va="top", ha="left")

    ax_text.text(0.0, 0.48, "Feedback:", fontsize=12, weight="bold", va="top", ha="left")
    ax_text.text(0.10, 0.48, feedback_wrapped, fontsize=11, va="top", ha="left")

    # BARS
    classes = list(top_series.index)[::-1]
    values  = list(top_series.values)[::-1]
    y = np.arange(len(classes))
    ax_bar.barh(y, values, color="#74a9cf", edgecolor="#3a6ea5")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(classes, fontsize=11)
    ax_bar.set_xlabel("Fixation %", fontsize=11)
    xmax = (max(values) * 1.25) if values else 1
    ax_bar.set_xlim(0, xmax)
    for i, v in enumerate(values):
        ax_bar.text(v + xmax*0.02, i, f"{v:.1f}%", va="center", fontsize=10)
    ax_bar.spines[['top','right']].set_visible(False)


def make_cards(pid, best_meta, worst_meta, best_row, worst_row,
               out_png: str, fig_w: float, fig_h: float, wrap_chars: int, max_text_lines: int):
    btop = best_row.sort_values(ascending=False).head(5)
    wtop = worst_row.sort_values(ascending=False).head(5)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
    outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1,1], wspace=0.20)

    _draw_card(fig, outer[0,0], "BEST",  best_meta,  btop, wrap_chars, max_text_lines)
    _draw_card(fig, outer[0,1], "WORST", worst_meta, wtop, wrap_chars, max_text_lines)

    fig.set_constrained_layout(True)
    fig.suptitle(f"{pid} – Segment cards (subjective + objective)", fontsize=16)
    plt.savefig(out_png, dpi=220)
    plt.close()
    return out_png


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="BEST vs WORST segment cards (no-overlap layout).")
    p.add_argument("--participant_id", required=True, help="e.g., P01")
    p.add_argument("--base_dir", default=DEFAULT_BASE)
    p.add_argument("--fig_w", type=float, default=15.0, help="Figure width (inches)")
    p.add_argument("--fig_h", type=float, default=8.0,  help="Figure height (inches)")
    p.add_argument("--wrap",  type=int,   default=72,   help="Wrap width (characters)")
    p.add_argument("--text_lines", type=int, default=6, help="Max feedback lines")
    args = p.parse_args()

    pid  = args.participant_id
    subj = load_subjective(args.base_dir, pid)
    objp = load_objective_percent(args.base_dir, pid)

    best, worst = pick_best_worst(subj)
    best_id, worst_id = int(best["segment_id"]), int(worst["segment_id"])

    def safe_row(seg_id):
        return objp.loc[seg_id] if seg_id in objp.index else pd.Series({c:0.0 for c in objp.columns})
    best_row, worst_row = safe_row(best_id), safe_row(worst_id)

    out_dir = os.path.join(args.base_dir, pid, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{pid}_best_worst_cards.png")

    best_meta = {
        "segment_id": best_id,
        "walkability_rating": float(best["walkability_rating"]),
        "themes": list(best.get("themes", [])),
        "feedback": str(best.get("feedback",""))
    }
    worst_meta = {
        "segment_id": worst_id,
        "walkability_rating": float(worst["walkability_rating"]),
        "themes": list(worst.get("themes", [])),
        "feedback": str(worst.get("feedback",""))
    }

    make_cards(pid, best_meta, worst_meta, best_row, worst_row,
               out_png, args.fig_w, args.fig_h, args.wrap, args.text_lines)

    print(f"✅ Saved: {out_png}")

if __name__ == "__main__":
    main()
