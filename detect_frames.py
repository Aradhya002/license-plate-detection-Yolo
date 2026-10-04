from pathlib import Path
from detect_plates import extract_frames, load_model, detect_plates, draw_detections
import cv2

VIDEO_PATH = "traffic_sample.mp4"   # put your video file here
MODEL_PATH = "best.pt"

model = load_model(MODEL_PATH)

# Step 1: break the video into individual frame images
frames = extract_frames(VIDEO_PATH, "frames", every_n_seconds=1.5)
print(f"Extracted {len(frames)} frames")

Path("plate_crops").mkdir(exist_ok=True)
total_plates = 0

# Step 2: run detection on each frame, one at a time
for frame_path in frames:
    dets = detect_plates(frame_path, model, conf_threshold=0.15)

    if not dets:
        continue  # no plates found in this frame, skip to next

    frame_name = Path(frame_path).stem  # e.g. "frame_000042"

    # Step 3: loop over every plate found IN this one frame
    for i, d in enumerate(dets):
        crop_filename = f"plate_crops/{frame_name}_plate{i}.jpg"
        cv2.imwrite(crop_filename, d["crop"])
        print(f"  {frame_name}: plate {i} -> confidence {d['confidence']}")
        total_plates += 1

print(f"\nDone. Found {total_plates} plate detection(s) across {len(frames)} frames.")