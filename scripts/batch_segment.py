import os
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# ✅ 1️⃣ Load model and processor
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-mapillary-vistas-semantic"
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-mapillary-vistas-semantic"
)

# ✅ 2️⃣ Set device (Force GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("❌ CUDA (GPU) not available. Aborting. You MUST run this on a GPU machine.")
model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")

# ✅ 3️⃣ Loop over participants
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))  # one level up from scripts/
DATA_DIR = os.path.join(BASE_DIR, "Data")

for pid in range(1, 9):
    participant_id = f"P{pid:02d}"
    video_path = os.path.join(DATA_DIR, participant_id, "raw", "Video.mp4")
    
    if not os.path.exists(video_path):
        print(f"\n🚶 Processing {participant_id}...\n❌ Video not found: {video_path}")
        continue

    print(f"\n🚶 Processing {participant_id}...")

    # Output folders
    out_base = os.path.join(DATA_DIR, participant_id, "semantic_segmentation")
    snapshots_dir = os.path.join(out_base, "snapshots")
    masks_dir = os.path.join(out_base, "masks")
    os.makedirs(snapshots_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"🎥 FPS: {fps:.2f} | Total Frames: {frame_count} | Duration: {duration:.2f}s")

    frame_idx = 0

    while cap.isOpened() and frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip if already processed
        mask_filename = os.path.join(masks_dir, f"mask_{frame_idx:05d}.png")
        if os.path.exists(mask_filename):
            frame_idx += 1
            continue

        # Convert BGR ➜ RGB ➜ PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        # Process input ➜ move to GPU
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process on CPU
        predicted_mask = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        # Save snapshot and mask
        image.save(os.path.join(snapshots_dir, f"frame_{frame_idx:05d}.jpg"))
        mask_pil = Image.fromarray(predicted_mask.byte().cpu().numpy())
        mask_pil.save(mask_filename)

        # Show progress
        percent = 100 * frame_idx / frame_count
        print(f"✅ {participant_id} - Frame {frame_idx + 1}/{frame_count} ({percent:.2f}%) done")

        frame_idx += 1

    cap.release()

print("\n🏁 ALL DONE. All videos processed 🚀")
