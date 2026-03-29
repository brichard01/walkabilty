#!/usr/bin/env python
# segment_pxx_mask2former.py
import os, argparse, cv2, torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

def load_model(device):
    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    ).to(device).eval()
    return processor, model

def main():
    ap = argparse.ArgumentParser(description="Run semantic segmentation for one participant video.")
    ap.add_argument("--participant_id", required=True, help="e.g. P09")
    ap.add_argument("--base_dir", required=True, help="e.g. C:/Abderrahim Internship Data/Data")
    ap.add_argument("--video", default="Video.mp4", help="video filename under raw/ (try scenevideo.mp4 if needed)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (>=1)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Run this on a GPU machine.")

    device = torch.device("cuda")
    processor, model = load_model(device)
    print(f"✅ Model on {device}")

    pid = args.participant_id
    base = os.path.abspath(args.base_dir)
    raw_dir = os.path.join(base, pid, "raw")

    # try the requested video name, then a common alternative
    video_path = os.path.join(raw_dir, args.video)
    if not os.path.exists(video_path):
        alt = os.path.join(raw_dir, "scenevideo.mp4")
        if os.path.exists(alt):
            video_path = alt
        else:
            raise FileNotFoundError(f"❌ No video found: {args.video} or scenevideo.mp4 in {raw_dir}")

    out_base = os.path.join(base, pid, "semantic_segmentation")
    snapshots_dir = os.path.join(out_base, "snapshots")
    masks_dir     = os.path.join(out_base, "masks")
    os.makedirs(snapshots_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 {pid}: {os.path.basename(video_path)} | FPS={fps:.2f} | Frames={frame_count}")

    idx = 0
    processed = 0
    while cap.isOpened() and idx < frame_count:
        ok, frame = cap.read()
        if not ok:
            break

        # skip frames if stride > 1
        if idx % args.stride != 0:
            idx += 1
            continue

        snap_path = os.path.join(snapshots_dir, f"frame_{idx:06d}.jpg")
        mask_path = os.path.join(masks_dir,     f"mask_{idx:06d}.png")
        if os.path.exists(mask_path):
            idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # target_sizes expects (H, W)
        pred = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_img.size[::-1]]
        )[0]  # (H, W) with class indices

        # save snapshot + mask
        pil_img.save(snap_path, quality=92)
        Image.fromarray(pred.byte().cpu().numpy()).save(mask_path)

        processed += 1
        if processed % 10 == 0:
            print(f"✅ {pid}: {processed} frames done (idx {idx}/{frame_count})")

        idx += 1

    cap.release()
    print(f"🏁 Done: {pid}. Snapshots ➜ {snapshots_dir} | Masks ➜ {masks_dir}")

if __name__ == "__main__":
    main()
