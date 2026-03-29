#!/usr/bin/env python
import argparse, os, ast, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- helpers --------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_segments_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    rows = []
    for s in y.get("segments", []):
        rows.append({
            "segment_id": int(s["segment_id"]),
            "walkability_rating": float(s["walkability_rating"]),
            "feedback": str(s.get("feedback","")).strip(),
            "themes": list(s.get("themes", [])),
        })
    return pd.DataFrame(rows)

def load_final(path):
    df = pd.read_csv(path)
    def parse_list(x):
        if isinstance(x, list): return x
        s = str(x)
        try:
            return ast.literal_eval(s)
        except:
            return [t.strip() for t in s.strip("[]").split(",")]
    df["top_classes"] = df["top_classes"].apply(parse_list)

    def clean_np_list(x):
        if isinstance(x, list): return [float(v) for v in x]
        s = str(x).replace("np.float64(","").replace(")","")
        return [float(v) for v in ast.literal_eval(s)]
    df["top_percentages"] = df["top_percentages"].apply(clean_np_list)
    return df

def derive_best_worst_table(base_dir, pid):
    """
    Return a tidy DataFrame with columns:
    class, best_pct, worst_pct, diff, best_seg, worst_seg
    """
    # subjective -> find best/worst
    subj = read_segments_yaml(os.path.join(base_dir, pid, f"segments_{pid}.yaml"))
    best_row = subj.sort_values(["walkability_rating","segment_id"], ascending=[False, True]).iloc[0]
    worst_row = subj.sort_values(["walkability_rating","segment_id"], ascending=[True, True]).iloc[0]
    best_seg, worst_seg = int(best_row["segment_id"]), int(worst_row["segment_id"])

    # objective -> read final table and pick the two segments
    fin = load_final(os.path.join(base_dir, pid, f"{pid}_final_analysis.csv"))

    try:
        brow = fin.loc[fin["segment_id"]==best_seg].iloc[0]
        wrow = fin.loc[fin["segment_id"]==worst_seg].iloc[0]
    except IndexError:
        # fallback to first rows if segment ids not found
        brow = fin.iloc[0]; wrow = fin.iloc[-1]

    b = pd.Series(dict(zip(brow["top_classes"], brow["top_percentages"])))
    w = pd.Series(dict(zip(wrow["top_classes"], wrow["top_percentages"])))
    df = pd.DataFrame({"best_pct": b, "worst_pct": w}).fillna(0.0).reset_index().rename(columns={"index":"class"})
    df["diff"] = df["best_pct"] - df["worst_pct"]
    df["best_seg"] = best_seg
    df["worst_seg"] = worst_seg
    return df.sort_values("diff", ascending=False)

# -------------------- plots --------------------
def plot_dumbbell(df_bw, pid, out_png, top_k=12):
    """
    A clean 'dumbbell' (dot+line) plot per participant:
        x: fixation %, y: class (top_k by max(best, worst))
        shows best vs worst and the difference visually
    """
    # choose classes
    df = df_bw.copy()
    df["max_val"] = df[["best_pct","worst_pct"]].max(axis=1)
    df = df.sort_values("max_val", ascending=True).tail(top_k)  # small->large so y asc
    y = np.arange(len(df))

    plt.figure(figsize=(10, 6))
    # stems
    for i, (b, w) in enumerate(zip(df["best_pct"], df["worst_pct"])):
        plt.plot([w, b], [y[i], y[i]], color="#c9c9c9", lw=2, zorder=1)

    # points
    plt.scatter(df["worst_pct"], y, color="#ffb703", edgecolor="#7a5a00", s=60, zorder=2, label="Worst segment")
    plt.scatter(df["best_pct"],  y, color="#219ebc", edgecolor="#0b486b", s=60, zorder=3, label="Best segment")

    plt.yticks(y, df["class"])
    plt.xlabel("Fixation %")
    title = f"{pid} — Best (seg {int(df['best_seg'].iloc[0])}) vs Worst (seg {int(df['worst_seg'].iloc[0])})"
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_diff_heatmap(best_worst_dict, out_png, top_m=15):
    """
    Heatmap of (best - worst) per participant per class.
    We keep the top_m classes by overall variance to avoid clutter.
    """
    # build wide matrix: rows=participants, cols=classes
    frames = []
    for pid, df in best_worst_dict.items():
        s = df.set_index("class")["diff"]
        frames.append(s.rename(pid))
    mat = pd.DataFrame(frames).fillna(0.0)

    # pick informative columns
    var_cols = mat.var(axis=0).sort_values(ascending=False).head(top_m).index
    mat_top = mat[var_cols]

    plt.figure(figsize=(max(8, 0.5*len(var_cols)+3), 0.6*len(mat_top)+2))
    sns.heatmap(mat_top, annot=True, fmt=".1f", cmap="RdBu_r", center=0, cbar_kws={"label": "Best − Worst (Fix %)"},
                linewidths=0.5, linecolor="#eaeaea", annot_kws={"fontsize":8})
    plt.title("Best − Worst fixation % by class (rows: participants)")
    plt.ylabel("Participant")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize per-participant best/worst CSVs + aggregate")
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--participants", nargs="+", required=True)  # P01 P02 ...
    ap.add_argument("--top_k", type=int, default=12, help="Top classes per participant (dumbbell)")
    ap.add_argument("--top_m", type=int, default=15, help="Top classes across participants (heatmap)")
    args = ap.parse_args()

    best_worst_dict = {}

    # 1) For each participant: ensure a tidy best/worst CSV and plot dumbbell
    for pid in args.participants:
        out_dir = os.path.join(args.base_dir, pid, "plots")
        ensure_dir(out_dir)

        # If you already have a CSV, load it; otherwise derive from final + yaml
        bw_csv = os.path.join(args.base_dir, pid, f"{pid}_best_worst.csv")
        if os.path.exists(bw_csv):
            df_bw = pd.read_csv(bw_csv)
        else:
            df_bw = derive_best_worst_table(args.base_dir, pid)
            df_bw.to_csv(bw_csv, index=False)

        # Dumbbell chart
        dumbbell_png = os.path.join(out_dir, f"{pid}_best_worst_dumbbell.png")
        plot_dumbbell(df_bw, pid, dumbbell_png, top_k=args.top_k)

        best_worst_dict[pid] = df_bw[["class","best_pct","worst_pct","diff"]].copy()

    # 2) Heatmap across participants (best - worst)
    agg_dir = os.path.join(args.base_dir, "_aggregate")
    ensure_dir(agg_dir)
    heatmap_png = os.path.join(agg_dir, "best_minus_worst_heatmap.png")
    plot_diff_heatmap(best_worst_dict, heatmap_png, top_m=args.top_m)

    print("✅ Done.")
    print(f"• Per participant dumbbells saved under …\\P0X\\plots\\")
    print(f"• Difference heatmap saved to: {heatmap_png}")

if __name__ == "__main__":
    main()
