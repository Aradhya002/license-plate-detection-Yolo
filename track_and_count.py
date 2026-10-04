"""
Count unique vehicles (by unique plate) passing through a camera's video,
using YOLO's built-in tracking (ByteTrack) instead of counting raw
per-frame detections - which would count the same car multiple times
since it stays visible across several frames.

Usage:
    python track_and_count.py
        (uses traffic_sample.mp4 by default)

    python track_and_count.py some_other_video.mp4
"""

import sys
from ultralytics import YOLO

MODEL_PATH = "best.pt"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "traffic_sample.mp4"


def main():
    model = YOLO(MODEL_PATH)

    # model.track() (not model.predict()) is what turns on ByteTrack.
    # It reads the video at its own native frame rate directly - it does
    # NOT use extract_frames()/every_n_seconds, because tracking needs
    # small, smooth movement between consecutive frames to follow an
    # object correctly; skipping frames would break that.
    # persist=True keeps IDs consistent across the whole run instead of
    # resetting them each frame. stream=True processes frame-by-frame
    # instead of loading the whole video into memory at once.
    results = model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",  # ships with ultralytics, nothing to install
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
            continue  # nothing detected/tracked in this frame
        for track_id in frame_result.boxes.id:
            seen_ids.add(int(track_id))

    print(f"Processed {frame_count} frames from {VIDEO_PATH}")
    print(f"Unique vehicles (plates) counted: {len(seen_ids)}")


if __name__ == "__main__":
    main()
