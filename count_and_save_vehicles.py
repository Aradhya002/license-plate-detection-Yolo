"""
Same idea as track_and_save_best.py, but for whole VEHICLES (car/motorcycle/
bus/truck) using the pretrained yolov8n.pt - saves one best crop per
tracked vehicle ID, so you can actually LOOK at what got counted instead
of just trusting the final number.

This is a diagnostic tool: if the count looks wrong, open vehicle_crops/
and check - are these genuinely different vehicles? The same vehicle
twice? Or something that isn't a vehicle at all (a sign, a shadow, a
tree) that got misclassified?

Usage:
    python count_and_save_vehicles.py
        (uses traffic_sample.mp4 by default)

    python count_and_save_vehicles.py some_other_video.mp4

Output:
    vehicle_crops/  - one image per tracked ID, named vehicle_<id>_<class>.jpg
"""

import sys
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "traffic_sample.mp4"
OUTPUT_DIR = Path("vehicle_crops")

# COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Ignore boxes shorter than this many pixels tall - farther-away vehicles
# appear smaller, so this is how you "limit the sight distance."
# Start with this value, then adjust based on results (see notes below).
MIN_BOX_HEIGHT = 80


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)

    results = model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        conf=0.35,
        persist=True,
        stream=True,
        verbose=False,
    )

    best_conf = {}
    best_crop = {}
    best_class = {}

    frame_count = 0
    for frame_result in results:
        frame_count += 1
        if frame_result.boxes is None or frame_result.boxes.id is None:
            continue

        frame_img = frame_result.orig_img

        for box in frame_result.boxes:
            track_id = int(box.id[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            box_height = y2 - y1
            if box_height < MIN_BOX_HEIGHT:
                continue  # too small = too far away, skip it

            crop = frame_img[y1:y2, x1:x2]

            if track_id not in best_conf or conf > best_conf[track_id]:
                best_conf[track_id] = conf
                best_crop[track_id] = crop
                best_class[track_id] = CLASS_NAMES.get(cls_id, f"class{cls_id}")

    for track_id, crop in best_crop.items():
        cls_name = best_class[track_id]
        out_path = OUTPUT_DIR / f"vehicle_{track_id}_{cls_name}.jpg"
        cv2.imwrite(str(out_path), crop)
        print(f"ID {track_id} ({cls_name}): confidence {best_conf[track_id]:.2f}")

    print(f"\nProcessed {frame_count} frames from {VIDEO_PATH}")
    print(f"Total tracked IDs: {len(best_crop)}")
    print(f"Crops saved to -> {OUTPUT_DIR}/  (open these and check them by eye)")


if __name__ == "__main__":
    main()
