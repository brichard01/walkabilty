import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, GroupKFold
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("train")


def metrics(y_true, y_pred):
    return dict(
        MAE=float(mean_absolute_error(y_true, y_pred)),
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        Spearman=float(spearmanr(y_true, y_pred).correlation),
    )


def folds_lopo(df: pd.DataFrame):
    for pid in df["participant_id"].unique():
        tr = df[df.participant_id != pid].reset_index(drop=True)
        te = df[df.participant_id == pid].reset_index(drop=True)
        yield f"LOPO_{pid}", tr, te


def folds_groupk(df: pd.DataFrame, n_splits=5, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["participant_id"].values
    for i, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=groups), 1):
        yield f"GroupK_{i}", df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


def folds_kfold(df: pd.DataFrame, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr_idx, te_idx) in enumerate(kf.split(df), 1):
        yield f"KFold_{i}", df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--cv", choices=["lopo", "groupk", "kfold"], default="lopo")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--features_csv", default="./outputs/features/segment_features_all.csv")
    args = ap.parse_args()

    feat_fp = Path(args.features_csv)
    assert feat_fp.exists(), f"Missing {feat_fp}"
    df = pd.read_csv(feat_fp)

    # keep only valid rows
    before = len(df)
    df = df[df["rating"].notna() & (df["segment_id"] != -1)].copy()
    log.info("Filtered rows: %d -> %d (dropped %d)", before, len(df), before - len(df))

    feature_cols = [c for c in df.columns if c not in {"participant_id", "segment_id", "rating"}]

    # choose folds
    if args.cv == "lopo":
        fold_gen = folds_lopo(df)
    elif args.cv == "groupk":
        fold_gen = folds_groupk(df, n_splits=args.n_splits)
    else:
        fold_gen = folds_kfold(df, n_splits=args.n_splits)

    rows = []
    for name, tr, te in fold_gen:
        Xtr, ytr = tr[feature_cols], tr["rating"].astype(float)
        Xte, yte = te[feature_cols], te["rating"].astype(float)

        # ----- Elastic-Net with explicit alpha grid (supports l1_ratio=0.0)
        enet = ElasticNetCV(
            alphas=np.logspace(-3, 1, 50),
            l1_ratio=[0.0, 0.2, 0.5, 0.8, 1.0],
            cv=5,
            random_state=42,
            n_jobs=-1,
        ).fit(Xtr, ytr)
        yhat_en = enet.predict(Xte)
        m_en = {f"enet_{k}": v for k, v in metrics(yte, yhat_en).items()}

        # ----- LightGBM (CPU-friendly)
        gbm = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        ).fit(Xtr, ytr)
        yhat_gb = gbm.predict(Xte)
        m_gb = {f"gbm_{k}": v for k, v in metrics(yte, yhat_gb).items()}

        rows.append({"fold": name, **m_en, **m_gb})

    res = pd.DataFrame(rows)
    out_dir = Path(args.out) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_fp = out_dir / "lopo_results.csv"
    res.to_csv(res_fp, index=False)
    log.info("Saved %s", res_fp)

    log.info(
        "Means — enet: MAE=%.3f RMSE=%.3f  ρ=%.3f | gbm: MAE=%.3f RMSE=%.3f  ρ=%.3f",
        res["enet_MAE"].mean(),
        res["enet_RMSE"].mean(),
        res["enet_Spearman"].mean(),
        res["gbm_MAE"].mean(),
        res["gbm_RMSE"].mean(),
        res["gbm_Spearman"].mean(),
    )

    # ----- train final models on ALL data (for SHAP/demo)
    X, y = df[feature_cols], df["rating"].astype(float)

    enet_all = ElasticNetCV(
        alphas=np.logspace(-3, 1, 50),
        l1_ratio=[0.0, 0.2, 0.5, 0.8, 1.0],
        cv=5,
        random_state=42,
        n_jobs=-1,
    ).fit(X, y)

    gbm_all = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    ).fit(X, y)

    joblib.dump({"enet": enet_all, "gbm": gbm_all, "features": feature_cols}, out_dir / "final_models.joblib")
    log.info("Saved %s", out_dir / "final_models.joblib")


if __name__ == "__main__":
    main()
