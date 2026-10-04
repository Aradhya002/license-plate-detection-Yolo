"""
Count ALL vehicles passing a camera - regardless of whether their plate
is visible or readable - by detecting and tracking the VEHICLE itself
(car/motorcycle/bus/truck), not the plate.

Uses the plain pretrained yolov8n.pt - NOT best.pt. It already knows these
vehicle classes from its original COCO training, so no retraining is
needed for this - it's a separate detector for a separate question.

This is for traffic density/counting. For "which specific vehicle is
this" (blacklist, trajectory), keep using best.pt + OCR as before - that
still needs the plate.

Usage:
    python count_all_vehicles.py
        (uses traffic_sample.mp4 by default)

    python count_all_vehicles.py some_other_video.mp4
"""

import sys
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"  # the ORIGINAL pretrained model, on purpose - not best.pt
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "traffic_sample.mp4"

# COCO class IDs this model already knows: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]


def main():
    model = YOLO(MODEL_PATH)

    results = model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,  # only track these, ignore people/signs/etc.
        conf=0.35,
        persist=True,
        stream=True,
        verbose=False,
    )

    seen_ids = set()
    frame_count = 0

    for frame_result in results:
        frame_count += 1
        if frame_result.boxes is None or frame_result.boxes.id is None:
            continue
        for track_id in frame_result.boxes.id:
            seen_ids.add(int(track_id))

    print(f"Processed {frame_count} frames from {VIDEO_PATH}")
    print(f"Total vehicles counted (car/motorcycle/bus/truck): {len(seen_ids)}")


if __name__ == "__main__":
    main()
