#!/usr/bin/env python
# build_best_worst_all.py
#
# One command to generate:
#   - {pid}/plots/segments/{pid}_best_worst_summary.csv
#   - {pid}/plots/segments/{pid}_best_worst_cards.png
#   - {pid}/plots/segments/{pid}_best_vs_worst_fixation_top12.png
#
# Works with:
#   - Subjective:   Data/Pxx/subjective/Pxx_subjective.csv  OR  Data/Pxx/segments_Pxx.yaml
#   - Objective %:  Data/Pxx/objective_analysis/Pxx_segment_fixation_percentage.csv
#
# Usage (Windows):
#   python build_best_worst_all.py --base_dir "C:/Abderrahim Internship Data/Data" --participants P01 P02 P03 P04 P05 P07 P08 --wrap 75 --fig_w 16 --fig_h 9 --topN 12

import argparse, os, ast, yaml, textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IRRELEVANT = {"Mountain","Sand","Snow","Boat"}
# Any "Unnamed" columns will be dropped
def _drop_junk_cols(df: pd.DataFrame) -> pd.DataFrame:
    junk = [c for c in df.columns if c.lower().startswith("unnamed")]
    return df.drop(columns=junk, errors="ignore")

# ---------- LOADERS ----------
def load_subjective(base: str, pid: str) -> pd.DataFrame:
    csv_path = os.path.join(base, pid, "subjective", f"{pid}_subjective.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8")
        df = _drop_junk_cols(df)
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
        if "themes" in df.columns:
            df["themes"] = df["themes"].apply(parse_themes)
        else:
            df["themes"] = [[]]*len(df)
        if "feedback" not in df.columns:
            df["feedback"] = ""
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
    df = pd.read_csv(f, encoding="utf-8")
    df = _drop_junk_cols(df)
    if "segment_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "segment_id"})
    df["segment_id"] = df["segment_id"].astype(int)
    df = df.set_index("segment_id")
    keep_cols = [c for c in df.columns if c not in IRRELEVANT and not c.lower().startswith("unnamed")]
    df = df[keep_cols].copy()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

# ---------- LOGIC ----------
def pick_best_worst(subj: pd.DataFrame):
    best = subj.sort_values(["walkability_rating","segment_id"], ascending=[False,True]).iloc[0]
    worst = subj.sort_values(["walkability_rating","segment_id"],  ascending=[True,True]).iloc[0]
    return best, worst

def wrap_text(s: str, width: int) -> str:
    s = str(s).replace("\n"," ").strip()
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False))

def topN_as_string(row: pd.Series, N=5):
    top = row.sort_values(ascending=False).head(N)
    return ", ".join([f"{k} ({v:.1f}%)" for k,v in top.items()])

# ---------- VISUALS ----------
def plot_cards(pid, best_meta, worst_meta, best_row, worst_row, outpath, fig_w=16, fig_h=9, wrap=80, text_lines=7):
    # Top 5 for each
    btop = best_row.sort_values(ascending=False).head(5)
    wtop = worst_row.sort_values(ascending=False).head(5)

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), height_ratios=[1.2, 1.0], width_ratios=[1,1])
    # Titles
    axes[0,0].axis('off'); axes[0,1].axis('off')

    title_l = f"BEST – Segment {best_meta['segment_id']} (Rating {float(best_meta['walkability_rating']):.1f})"
    title_r = f"WORST – Segment {worst_meta['segment_id']} (Rating {float(worst_meta['walkability_rating']):.1f})"
    axes[0,0].text(0, 0.9, title_l, fontsize=13, weight='bold', va='top')
    axes[0,1].text(0, 0.9, title_r, fontsize=13, weight='bold', va='top')

    # Themes + Feedback (wrapped)
    axes[0,0].text(0, 0.68, "Themes:", fontsize=11, weight='bold', va='top')
    axes[0,0].text(0, 0.62, wrap_text(", ".join(best_meta.get("themes", [])), wrap), fontsize=10, va='top')

    axes[0,0].text(0, 0.44, "Feedback:", fontsize=11, weight='bold', va='top')
    axes[0,0].text(0, 0.38, "\n".join(wrap_text(best_meta.get("feedback",""), wrap).split("\n")[:text_lines]), fontsize=10, va='top')

    axes[0,1].text(0, 0.68, "Themes:", fontsize=11, weight='bold', va='top')
    axes[0,1].text(0, 0.62, wrap_text(", ".join(worst_meta.get("themes", [])), wrap), fontsize=10, va='top')

    axes[0,1].text(0, 0.44, "Feedback:", fontsize=11, weight='bold', va='top')
    axes[0,1].text(0, 0.38, "\n".join(wrap_text(worst_meta.get("feedback",""), wrap).split("\n")[:text_lines]), fontsize=10, va='top')

    # Horizontal bars (bottom row)
    def draw_bar(ax, series):
        classes = list(series.index)[::-1]
        values  = list(series.values)[::-1]
        ax.barh(np.arange(len(classes)), values, color='#79a9e8', edgecolor='#3d6fb3')
        ax.set_yticks(np.arange(len(classes))); ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel("Fixation %", fontsize=10)
        xmax = (max(values) * 1.25) if values else 1
        ax.set_xlim(0, xmax)
        for i, v in enumerate(values):
            ax.text(v + xmax*0.015, i, f"{v:.1f}%", va='center', fontsize=9)
        for s in ["top","right"]:
            ax.spines[s].set_visible(False)

    draw_bar(axes[1,0], btop)
    draw_bar(axes[1,1], wtop)

    fig.suptitle(f"{pid} – Segment cards (subjective + objective)", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(outpath, dpi=240)
    plt.close()

def plot_fixation_compare(pid, best_id, worst_id, best_row, worst_row, outpath, topN=12, fig_w=14, fig_h=6):
    mean_series = ((best_row + worst_row)/2.0).sort_values(ascending=False)
    keep = mean_series.index[:topN].tolist()

    best_vals = best_row[keep].values
    worst_vals = worst_row[keep].values

    x = np.arange(len(keep))
    w = 0.42
    plt.figure(figsize=(fig_w, fig_h))
    plt.bar(x - w/2, best_vals, width=w, label=f"Best seg {best_id}", edgecolor='black')
    plt.bar(x + w/2, worst_vals, width=w, label=f"Worst seg {worst_id}", edgecolor='black')
    plt.xticks(x, keep, rotation=45, ha='right')
    plt.ylabel("Fixation time (%)")
    plt.title(f"{pid} – Fixation % by class (Best vs Worst segment)")
    ymax = (max(best_vals.max() if best_vals.size else 0, worst_vals.max() if worst_vals.size else 0)*1.2) or 10
    plt.ylim(0, ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()

# ---------- CSV SUMMARY ----------
def write_summary_csv(out_csv, best_meta, worst_meta, best_row, worst_row):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    rows.append({
        "which":"best",
        "segment_id": int(best_meta["segment_id"]),
        "rating": float(best_meta["walkability_rating"]),
        "themes": "; ".join(best_meta.get("themes", [])),
        "feedback": str(best_meta.get("feedback","")).replace("\n"," ").strip(),
        "top_classes": topN_as_string(best_row, 5)
    })
    rows.append({
        "which":"worst",
        "segment_id": int(worst_meta["segment_id"]),
        "rating": float(worst_meta["walkability_rating"]),
        "themes": "; ".join(worst_meta.get("themes", [])),
        "feedback": str(worst_meta.get("feedback","")).replace("\n"," ").strip(),
        "top_classes": topN_as_string(worst_row, 5)
    })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# ---------- MAIN ----------
def process_one(base, pid, fig_w, fig_h, wrap, text_lines, topN):
    subj = load_subjective(base, pid)
    objp = load_objective_percent(base, pid)

    best, worst = pick_best_worst(subj)
    best_id, worst_id = int(best["segment_id"]), int(worst["segment_id"])

    def safe_row(seg_id):
        return objp.loc[seg_id] if seg_id in objp.index else pd.Series({c:0.0 for c in objp.columns})
    best_row = safe_row(best_id)
    worst_row = safe_row(worst_id)

    out_dir = os.path.join(base, pid, "plots", "segments")
    os.makedirs(out_dir, exist_ok=True)

    # CSV summary
    summary_csv = os.path.join(out_dir, f"{pid}_best_worst_summary.csv")
    write_summary_csv(summary_csv, best, worst, best_row, worst_row)

    # Cards
    cards_png = os.path.join(out_dir, f"{pid}_best_worst_cards.png")
    plot_cards(pid,
               {"segment_id": best_id, "walkability_rating": best["walkability_rating"], "themes": best.get("themes", []), "feedback": best.get("feedback","")},
               {"segment_id": worst_id,"walkability_rating": worst["walkability_rating"],"themes": worst.get("themes", []),"feedback": worst.get("feedback","")},
               best_row, worst_row,
               cards_png, fig_w=fig_w, fig_h=fig_h, wrap=wrap, text_lines=text_lines)

    # Fixation compare
    top12_png = os.path.join(out_dir, f"{pid}_best_vs_worst_fixation_top{topN}.png")
    plot_fixation_compare(pid, best_id, worst_id, best_row, worst_row, top12_png, topN=topN)

    print(f"✅ {pid}: wrote\n  - {summary_csv}\n  - {cards_png}\n  - {top12_png}")
    return summary_csv

def main():
    ap = argparse.ArgumentParser(description="Build best/worst summaries + visuals for multiple participants.")
    ap.add_argument("--base_dir", required=True, help="Base data dir, e.g. C:/Abderrahim Internship Data/Data")
    ap.add_argument("--participants", nargs="+", required=True, help="List like P01 P02 ...")
    ap.add_argument("--fig_w", type=float, default=16.0)
    ap.add_argument("--fig_h", type=float, default=9.0)
    ap.add_argument("--wrap", type=int, default=80)
    ap.add_argument("--text_lines", type=int, default=7)
    ap.add_argument("--topN", type=int, default=12, help="Top-N classes to compare in bar chart")
    args = ap.parse_args()

    all_csvs = []
    for pid in args.participants:
        try:
            csv_path = process_one(args.base_dir, pid, args.fig_w, args.fig_h, args.wrap, args.text_lines, args.topN)
            all_csvs.append((pid, csv_path))
        except Exception as e:
            print(f"❌ {pid}: {e}")

    # Optional: aggregate a quick index CSV listing where each participant's summary is
    if all_csvs:
        idx = pd.DataFrame(all_csvs, columns=["participant_id","summary_csv"])
        agg_dir = os.path.join(args.base_dir, "aggregate")
        os.makedirs(agg_dir, exist_ok=True)
        idx.to_csv(os.path.join(agg_dir, "best_worst_summary_index.csv"), index=False)
        print(f"\n📁 Index written: {os.path.join(agg_dir, 'best_worst_summary_index.csv')}")

if __name__ == "__main__":
    main()
