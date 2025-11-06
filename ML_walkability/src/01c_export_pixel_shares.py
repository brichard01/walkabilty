import argparse, re, logging
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

from utils.class_map import CLASS_ID_TO_LABEL
from utils.supercats import SUPER_CATS

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("pixelshares")

SUPER = ["greenery","ped_infra","people_bike","motor_traffic",
         "roadway","buildings","furniture_signage","sky_water"]

def to_supercat(class_id: int):
    label = CLASS_ID_TO_LABEL.get(int(class_id))
    if not label:
        return None
    for sc, members in SUPER_CATS.items():
        if label in members:
            return sc
    return None

def index_masks(mask_dir: Path):
    """Build frame_idx -> file map by parsing any number in the filename stem."""
    idx = {}
    for p in mask_dir.glob("*.png"):
        m = re.search(r"(\d+)", p.stem)
        if m:
            idx[int(m.group(1))] = p
    return idx

def frame_hist(mask_path: Path, max_id=65):
    """Return (counts per class_id, total_pixels) for a mask image."""
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:  # if RGB/palette-with-explicit channels, use first channel
        arr = arr[..., 0]
    counts = np.bincount(arr.ravel().astype(np.int64), minlength=max_id)
    return counts, int(arr.size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--max_frames_per_segment", type=int, default=60)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_fp = Path(args.out) / "features" / "segment_pixel_shares.csv"
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    parts = [d for d in data_root.iterdir() if d.is_dir() and d.name.upper().startswith("P")]
    for pdir in sorted(parts, key=lambda x: x.name):
        pid = pdir.name

        # --- gaze csv (must have frame_idx + segment_id)
        gaze_csvs = [p for p in pdir.glob("*_gaze_with_segment.csv") if not p.name.startswith(".")]
        
        if not gaze_csvs:
            log.warning("[%s] no *_gaze_with_segment.csv, skipping", pid)
            continue
        gaze = pd.read_csv(gaze_csvs[0])
        if "segment_id" not in gaze.columns or "frame_idx" not in gaze.columns:
            log.warning("[%s] gaze file missing segment_id/frame_idx, skipping", pid)
            continue

        # --- masks folder (your structure uses semantic_segmentation/masks)
        mask_dir = pdir / "semantic_segmentation" / "masks"
        if not mask_dir.exists():
            # fallback if someone copied masks directly under Pxx/masks
            alt = pdir / "masks"
            if alt.exists():
                mask_dir = alt
        if not mask_dir.exists():
            log.warning("[%s] no masks folder found at %s, skipping", pid, mask_dir)
            continue

        fmap = index_masks(mask_dir)
        if not fmap:
            log.warning("[%s] no PNG masks with numeric names in %s, skipping", pid, mask_dir)
            continue

        # --- per segment aggregation
        for seg_id, grp in gaze.groupby("segment_id"):
            if int(seg_id) == -1:
                continue
            frames = sorted(set(grp["frame_idx"].astype(int)))
            frames = [f for f in frames if f in fmap]
            if not frames:
                continue

            # sample up to N frames for speed; fixed seed for reproducibility
            if len(frames) > args.max_frames_per_segment:
                rng = np.random.default_rng(42)
                frames = sorted(rng.choice(frames, size=args.max_frames_per_segment, replace=False).tolist())

            total_counts = None
            total_pixels = 0
            for f in frames:
                c, n = frame_hist(fmap[f])
                total_counts = c if total_counts is None else (total_counts + c)
                total_pixels += n

            # map class counts -> super-category counts
            per_sc = {sc: 0.0 for sc in SUPER}
            for cid, cnt in enumerate(total_counts):
                if cnt == 0:
                    continue
                sc = to_supercat(cid)
                if sc:
                    per_sc[sc] += float(cnt)

            # normalize to shares
            if total_pixels > 0:
                for sc in SUPER:
                    per_sc[sc] = per_sc[sc] / float(total_pixels)

            rows.append({
                "participant_id": pid,
                "segment_id": int(seg_id),
                **{f"pixel_{k}": v for k, v in per_sc.items()}
            })

        n_for_pid = sum(1 for r in rows if r["participant_id"] == pid)
        log.info("[ok] %s -> %d segments", pid, n_for_pid)

    out = pd.DataFrame(rows)
    out.to_csv(out_fp, index=False)
    log.info("Wrote %s (%d rows)", out_fp, len(out))

if __name__ == "__main__":
    main()
