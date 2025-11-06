import argparse, logging
from pathlib import Path
import pandas as pd
import yaml

from utils.class_map import CLASS_ID_TO_LABEL
from utils.supercats import SUPER_CATS, IGNORE_IDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("build_features")

SUPER_LIST = ["greenery","ped_infra","people_bike","motor_traffic","roadway",
              "buildings","furniture_signage","sky_water"]

def label_to_supercat(label: str):
    for sc, members in SUPER_CATS.items():
        if label in members:
            return sc
    return None

def read_gaze_file(p_dir: Path) -> pd.DataFrame:
    # Prefer *_gaze_with_segment.csv (has segment_id). Adjust the pattern if needed.
    cands = [p for p in p_dir.glob("*_gaze_with_segment.csv") if not p.name.startswith(".")]

    assert cands, f"No gaze_with_segment CSV found in {p_dir}"
    fp = cands[0]
    df = pd.read_csv(fp)
    required = {"frame_idx","timestamp_us","fixation_x","fixation_y",
                "fixation_duration_sec","class_id","segment_id"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns in {fp}: {missing}"
    return df

def read_ratings(p_dir: Path) -> pd.DataFrame:
    fp = p_dir / "ratings.csv"
    assert fp.exists(), f"Missing ratings.csv in {p_dir}"
    y = pd.read_csv(fp)
    assert {"segment_id","rating"}.issubset(y.columns), "ratings.csv must have segment_id,rating"
    return y

def features_for_participant(p_dir: Path) -> pd.DataFrame:
    pid = p_dir.name
    df = read_gaze_file(p_dir)

    # Map class IDs to labels
    df["class_label"] = df["class_id"].map(CLASS_ID_TO_LABEL)

    # Drop ignored IDs and rows with unknown labels
    df = df[~df["class_id"].isin(IGNORE_IDS)].copy()
    df = df.dropna(subset=["class_label"]).copy()

    # Map to super-categories
    df["supercat"] = df["class_label"].apply(label_to_supercat)
    unknown = df["supercat"].isna().sum()
    if unknown:
        log.warning("[%s] dropping %d fixations with unmapped labels", pid, unknown)
    df = df.dropna(subset=["supercat"]).copy()

    # Aggregate fixation durations (seconds) per (segment, supercat)
    g = df.groupby(["segment_id","supercat"], as_index=False)["fixation_duration_sec"].sum()
    tot = g.groupby("segment_id", as_index=False)["fixation_duration_sec"].sum().rename(
        columns={"fixation_duration_sec":"total_sec"})
    g = g.merge(tot, on="segment_id")
    g["fix_share"] = g["fixation_duration_sec"] / g["total_sec"]

    # Pivot to wide (one row per segment)
    feat = g.pivot(index="segment_id", columns="supercat", values="fix_share").fillna(0.0).reset_index()

    # Gaze stats per segment
    stats = df.groupby("segment_id", as_index=False).agg(
        n_fix=("fixation_duration_sec","count"),
        mean_fix_sec=("fixation_duration_sec","mean"),
        total_fix_sec=("fixation_duration_sec","sum"),
    )
    feat = feat.merge(stats, on="segment_id", how="left")

    # Ratings
    y = read_ratings(p_dir)
    feat = feat.merge(y, on="segment_id", how="left")

    # Ensure all supercat columns exist
    for sc in SUPER_LIST:
        if sc not in feat.columns:
            feat[sc] = 0.0

    # Add participant_id column
    feat.insert(0, "participant_id", pid)

    # Keep column order tidy
    cols = ["participant_id","segment_id","rating"] + SUPER_LIST + ["n_fix","mean_fix_sec","total_fix_sec"]
    feat = feat[cols]
    return feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data", help="Folder containing P01..PN folders")
    ap.add_argument("--out", default="./outputs", help="Where to write features CSV")
    ap.add_argument("--min_total_sec", type=float, default=None, help="Drop segments with total fixation below this (sec)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.out) / "features"
    out.mkdir(parents=True, exist_ok=True)

    parts = [d for d in data_root.iterdir() if d.is_dir() and d.name.upper().startswith("P")]
    assert parts, f"No participant folders found in {data_root}"

    frames = []
    for p in sorted(parts, key=lambda x: x.name):
        try:
            f = features_for_participant(p)
            frames.append(f)
            log.info("[ok] %s -> %d segments", p.name, len(f))
        except AssertionError as e:
            log.error("[skip] %s: %s", p.name, e)

    df = pd.concat(frames, ignore_index=True)

    # Optional filter: drop very short-total-fixation segments
    if args.min_total_sec is not None:
        before = len(df)
        df = df[df["total_fix_sec"] >= float(args.min_total_sec)].copy()
        log.info("Dropped %d segments with total_fix_sec < %.3f", before - len(df), args.min_total_sec)

    out_fp = out / "segment_features.csv"
    df.to_csv(out_fp, index=False)
    log.info("Wrote %s (%d rows)", out_fp, len(df))

if __name__ == "__main__":
    main()
