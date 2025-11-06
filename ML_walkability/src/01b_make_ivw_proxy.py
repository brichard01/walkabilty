import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("ivw_proxy")

REQUIRED_PIXEL_COLS = [
    "pixel_greenery", "pixel_ped_infra", "pixel_people_bike", "pixel_motor_traffic",
    "pixel_roadway", "pixel_buildings", "pixel_furniture_signage", "pixel_sky_water"
]

def clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="Path to outputs/features/segment_pixel_shares.csv")
    ap.add_argument("--out", dest="out", required=True,
                    help="Where to write outputs/features/ivw_all_proxy.csv")
    # optional knobs if you ever want to tweak the proxy later:
    ap.add_argument("--w_greenery", type=float, default=0.35)
    ap.add_argument("--w_pedinfra", type=float, default=0.35)
    ap.add_argument("--w_enclosure", type=float, default=0.25)
    ap.add_argument("--w_traffic", type=float, default=0.25)
    ap.add_argument("--bonus_sky_in_green", type=float, default=0.15,
                    help="small bonus from sky/water to greenery proxy")
    ap.add_argument("--penalize_sky_in_enclosure", type=float, default=0.10,
                    help="subtract some sky from enclosure proxy")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    assert {"participant_id", "segment_id"}.issubset(df.columns), \
        "Input must have participant_id and segment_id columns"

    # Ensure pixel columns exist; if any missing, fill with zeros
    for c in REQUIRED_PIXEL_COLS:
        if c not in df.columns:
            log.warning("Missing column %s in %s — filling with zeros.", c, inp)
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    # ---- Proxies for IVW dimensions (0..1)
    greenery = df["pixel_greenery"]
    sky = df["pixel_sky_water"]
    ped = df["pixel_ped_infra"]
    bld = df["pixel_buildings"]
    traf = df["pixel_motor_traffic"] + df["pixel_roadway"]

    ivw_greenery_proxy = clip01(greenery + args.bonus_sky_in_green * sky)
    ivw_pedinfra_proxy = clip01(ped)
    ivw_enclosure_proxy = clip01(bld - args.penalize_sky_in_enclosure * sky)
    ivw_traffic_proxy = clip01(traf)

    # ---- Combine into a single IVW-like score (normative: traffic negative)
    raw = (
        args.w_greenery * ivw_greenery_proxy
        + args.w_pedinfra * ivw_pedinfra_proxy
        + args.w_enclosure * ivw_enclosure_proxy
        - args.w_traffic * ivw_traffic_proxy
    )

    # Min–max scale raw to 0..1 for comparability across datasets
    rmin, rmax = float(raw.min()), float(raw.max())
    eps = 1e-9
    if rmax - rmin < eps:
        ivw_total_proxy = pd.Series(np.zeros(len(raw)), index=raw.index)
        log.warning("Raw IVW proxy had near-constant values; total proxy set to zeros.")
    else:
        ivw_total_proxy = (raw - rmin) / (rmax - rmin)

    out = pd.DataFrame({
        "participant_id": df["participant_id"],
        "segment_id": df["segment_id"].astype(int),
        "ivw_total_proxy": clip01(ivw_total_proxy),
        "ivw_greenery_proxy": ivw_greenery_proxy,
        "ivw_pedinfra_proxy": ivw_pedinfra_proxy,
        "ivw_enclosure_proxy": ivw_enclosure_proxy,
        "ivw_traffic_proxy": ivw_traffic_proxy
    })

    out.to_csv(outp, index=False)
    log.info("Wrote %s (%d rows) with columns: %s",
             outp, len(out), [c for c in out.columns if c.startswith("ivw")])

if __name__ == "__main__":
    main()
