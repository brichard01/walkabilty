import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SUPER = ["greenery","ped_infra","people_bike","motor_traffic",
         "roadway","buildings","furniture_signage","sky_water"]

def compute_alignment(row):
    eps = 1e-9
    g = np.array([row.get(sc, 0.0) for sc in SUPER], dtype=float)
    p = np.array([row.get(f"pixel_{sc}", 0.0) for sc in SUPER], dtype=float)

    g = g / (g.sum() + eps)
    p = p / (p.sum() + eps)

    # L1 alignment (0..1): 1 - 0.5 * L1 distance
    l1_gap = np.sum(np.abs(g - p))
    align_l1 = 1.0 - 0.5 * l1_gap

    # KL divergence (gaze || pixel)
    kl = np.sum(g * (np.log(g + eps) - np.log(p + eps)))

    # Rank correlation
    rho = spearmanr(g, p).correlation
    if np.isnan(rho):
        rho = 0.0

    return pd.Series({"align_l1": align_l1, "kl_gaze_pixel": kl, "rho_gaze_pixel": float(rho)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True, help="outputs/features/segment_features.csv")
    ap.add_argument("--pixel_csv", required=True, help="outputs/features/segment_pixel_shares.csv")
    ap.add_argument("--ivw_csv", required=False, help="outputs/features/ivw_all_proxy.csv or real IVW")
    ap.add_argument("--out_csv", default="./outputs/features/segment_features_all.csv")
    args = ap.parse_args()

    feat = pd.read_csv(args.features_csv)   # gaze shares + gaze stats + rating
    pix  = pd.read_csv(args.pixel_csv)      # pixel_* shares

    # merge gaze & pixels
    df = feat.merge(pix, on=["participant_id","segment_id"], how="left")

    # optionally merge IVW (proxy or official)
    if args.ivw_csv:
        ivw = pd.read_csv(args.ivw_csv)
        df = df.merge(ivw, on=["participant_id","segment_id"], how="left")

    # fill missing pixel_* with 0 so alignment works
    for sc in SUPER:
        col = f"pixel_{sc}"
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # compute alignment metrics
    align = df.apply(compute_alignment, axis=1)
    df = pd.concat([df, align], axis=1)

    out_fp = Path(args.out_csv)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)
    print(f"[ok] wrote {out_fp} with pixel_* + align_* "
          f"{'(and IVW columns)' if args.ivw_csv else '(no IVW provided)'}")

if __name__ == "__main__":
    main()
