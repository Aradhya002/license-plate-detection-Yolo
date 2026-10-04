"""
ANPR - License Plate Detection (YOLO side)
Owner: Member 1 (you)

This module owns detect_plates(). Your teammate's read_plate() (OCR) takes
the "crop" this returns and outputs plate text - keep that contract stable
so you two can work in parallel without blocking each other.

Output contract (this is what Member 4 will ingest into the DB, so don't
change the key names without telling the backend team):

    {
        "bbox": [x1, y1, x2, y2],   # pixel coords, ints
        "confidence": float,        # 0-1
        "crop": np.ndarray          # BGR cropped plate region, for OCR input
    }
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

# Swap this for your fine-tuned plate weights once you have them
# (e.g. "runs/detect/train/weights/best.pt"). yolov8n.pt is COCO-pretrained
# and does NOT know what a license plate is - it's only here so the pipeline
# runs end-to-end on Day 1 before you have real weights.
DEFAULT_MODEL_PATH = "yolov8n.pt"


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    """Load a YOLO model from a local checkpoint or an ultralytics-hosted name."""
    return YOLO(model_path)


def detect_plates(image, model: YOLO, conf_threshold: float = 0.35):
    """
    Run plate detection on a single image.

    Args:
        image: path to an image file, OR a numpy array (BGR, e.g. a video frame).
        model: a loaded YOLO model (see load_model()).
        conf_threshold: minimum confidence to keep a detection. Start around
            0.3-0.4 and tune once you see real false-positive/negative rates.

    Returns:
        List of detection dicts (see module docstring for the schema).
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        img = image

    results = model.predict(source=img, conf=conf_threshold, verbose=False)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 4),
                "crop": img[y1:y2, x1:x2],
            })
    return detections


def draw_detections(image, detections, out_path: str | None = None):
    """Draw bboxes on the image so you can sanity-check results by eye."""
    img = cv2.imread(str(image)).copy() if isinstance(image, (str, Path)) else image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{det["confidence"]:.2f}', (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if out_path:
        cv2.imwrite(str(out_path), img)
    return img


def extract_frames(video_path: str, out_dir: str, every_n_seconds: float = 1.5):
    """
    Pull frames out of a traffic video for detection - the plan's Day 4 task
    ("extract frames every 1-2 seconds"). Returns the list of saved frame paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(int(fps * every_n_seconds), 1)

    saved, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            path = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(str(path))
        idx += 1
    cap.release()
    return saved


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_PATH

    model = load_model(model_path)
    dets = detect_plates(image_path, model)

    print(f"Found {len(dets)} plate(s):")
    for i, d in enumerate(dets):
        print(f"  [{i}] bbox={d['bbox']}  confidence={d['confidence']}")

    draw_detections(image_path, dets, out_path="detections_preview.jpg")
    print("Saved annotated preview -> detections_preview.jpg")
