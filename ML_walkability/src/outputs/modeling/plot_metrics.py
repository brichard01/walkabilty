import os
import pandas as pd
import matplotlib.pyplot as plt

# ---- paths ----
RESULTS_CSV = "outputs/lopo_results.csv"
FIG_DIR     = "outputs/figures"
SUMMARY_CSV = os.path.join(FIG_DIR, "metrics_summary.csv")
BAR_PNG     = os.path.join(FIG_DIR, "metrics_bar.png")

# ---- ensure figure dir exists ----
os.makedirs(FIG_DIR, exist_ok=True)

# ---- load results ----
m = pd.read_csv(RESULTS_CSV)

# Keep only expected columns; drop rows with missing metrics
needed = ["model", "RMSE", "R2", "Spearman"]
m = m[needed].dropna()

# ---- aggregate mean/std per model ----
summary = (
    m.groupby("model")[["RMSE", "R2", "Spearman"]]
     .agg(["mean", "std"])
     .sort_values(("RMSE", "mean"))          # sort best (lower RMSE) first
     .round(3)
)

# write summary table
summary.to_csv(SUMMARY_CSV, index=True)

# ---- bar plot: RMSE by model with std as error bars ----
ax = summary[("RMSE", "mean")].plot(
    kind="bar",
    yerr=summary[("RMSE", "std")],
    capsize=4,
    rot=0
)
ax.set_ylabel("RMSE")
ax.set_title("LOPO performance by model")
ax.set_xlabel("")  # cleaner

plt.tight_layout()
plt.savefig(BAR_PNG, dpi=200)
plt.close()

print(f"[ok] wrote: {SUMMARY_CSV}")
print(f"[ok] wrote: {BAR_PNG}")
