import argparse, logging
from pathlib import Path
import joblib, pandas as pd, numpy as np
import shap, matplotlib.pyplot as plt
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("shap")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--features_csv", default="./outputs/features/segment_features_all.csv")
    ap.add_argument("--sample_n", type=int, default=1000, help="rows to sample for SHAP (keeps memory low)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    feat_fp = Path(args.features_csv)
    model_fp = out_dir / "models" / "final_models.joblib"
    fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    # load data + model
    df = pd.read_csv(feat_fp)
    bundle = joblib.load(model_fp)
    model: lgb.LGBMRegressor = bundle["gbm"]
    feat_cols = bundle["features"]

    # keep valid rows (same filter as training)
    df = df[df["rating"].notna() & (df["segment_id"] != -1)].copy()
    X = df[feat_cols]; y = df["rating"].astype(float)

    # sample for SHAP to avoid memory spikes
    n = min(len(X), args.sample_n)
    Xs = X.sample(n=n, random_state=42) if len(X) > n else X.copy()
    log.info("Using %d rows for SHAP (of %d total)", len(Xs), len(X))

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)

    # 1) global beeswarm
    plt.figure()
    shap.summary_plot(sv, Xs, show=False)
    plt.tight_layout()
    beeswarm_fp = fig_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_fp, dpi=200)
    plt.close()
    log.info("Saved %s", beeswarm_fp)

    # 2) global bar (mean |SHAP|)
    plt.figure()
    shap.summary_plot(sv, Xs, plot_type="bar", show=False)
    plt.tight_layout()
    bar_fp = fig_dir / "shap_bar_mean_abs.png"
    plt.savefig(bar_fp, dpi=200)
    plt.close()
    log.info("Saved %s", bar_fp)

    # 3) export mean |SHAP| as CSV (nice for thesis table)
    mean_abs = pd.DataFrame({
        "feature": Xs.columns,
        "mean_abs_shap": np.abs(sv).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    mean_abs_fp = fig_dir / "shap_mean_abs.csv"
    mean_abs.to_csv(mean_abs_fp, index=False)
    log.info("Saved %s", mean_abs_fp)

    # 4) fallback: model's built-in feature importance (gain)
    importances = pd.DataFrame({
        "feature": X.columns,
        "gain_importance": model.booster_.feature_importance(importance_type="gain")
    }).sort_values("gain_importance", ascending=False)
    imp_fp = fig_dir / "lgbm_feature_importance_gain.csv"
    importances.to_csv(imp_fp, index=False)
    log.info("Saved %s", imp_fp)

    # 5) one example waterfall (pick the row with highest residual)
    try:
        y_pred = model.predict(X)
        residual = np.abs(y - y_pred)
        idx = int(residual.idxmax())
        row = X.iloc[[idx]]

        sv_row = explainer.shap_values(row)
        plt.figure()
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, sv_row[0], row.iloc[0], show=False)
        plt.tight_layout()
        wf_fp = fig_dir / "shap_waterfall_example.png"
        plt.savefig(wf_fp, dpi=200)
        plt.close()
        log.info("Saved %s (example segment_id=%s, participant=%s, true=%.2f, pred=%.2f)",
                 wf_fp, df.iloc[idx]["segment_id"], df.iloc[idx]["participant_id"], y.iloc[idx], y_pred[idx])
    except Exception as e:
        log.warning("Skipping waterfall example due to: %s", e)

if __name__ == "__main__":
    main()
