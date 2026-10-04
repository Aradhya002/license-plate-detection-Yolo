"""
Track vehicles across a video, and save just ONE clean crop per unique
vehicle (by track ID) - the sharpest one (highest confidence) - instead
of a crop for every single frame that vehicle appeared in.

Usage:
    python track_and_save_best.py
        (uses traffic_sample.mp4 by default)

    python track_and_save_best.py some_other_video.mp4

Output:
    best_crops/  - one image per unique vehicle, named vehicle_<id>.jpg
"""

import sys
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "best.pt"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "traffic_sample.mp4"
OUTPUT_DIR = Path("best_crops")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)

    results = model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",
        conf=0.35,
        persist=True,
        stream=True,
        verbose=False,
    )

    best_conf = {}   # track_id -> best confidence seen so far for that vehicle
    best_crop = {}   # track_id -> the crop image that earned that best confidence

    frame_count = 0
    for frame_result in results:
        frame_count += 1
        if frame_result.boxes is None or frame_result.boxes.id is None:
            continue  # nothing tracked in this frame

        frame_img = frame_result.orig_img  # the raw frame, as an image array

        for box in frame_result.boxes:
            track_id = int(box.id[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            crop = frame_img[y1:y2, x1:x2]

            # Keep this crop only if it's this vehicle's best confidence so far
            if track_id not in best_conf or conf > best_conf[track_id]:
                best_conf[track_id] = conf
                best_crop[track_id] = crop

    for track_id, crop in best_crop.items():
        out_path = OUTPUT_DIR / f"vehicle_{track_id}.jpg"
        cv2.imwrite(str(out_path), crop)
        print(f"Vehicle {track_id}: saved best crop (confidence {best_conf[track_id]:.2f})")

    print(f"\nProcessed {frame_count} frames from {VIDEO_PATH}")
    print(f"Unique vehicles: {len(best_crop)}")
    print(f"Best crops saved to -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
