# ANPR - YOLO Plate Detection (your part of Member 1)

You're doing detection (find the plate in the frame → bbox + crop). Your
teammate does OCR (crop → text). The handoff between you is the `crop` field
in each detection dict from `detect_plates()` - agree on that early and you
can both work independently.

## 1. Environment setup (Day 1)

```bash
pip install -r requirements.txt
```

That's `ultralytics` (YOLOv8), `opencv-python`, `numpy`. Ultralytics will
auto-download the base `yolov8n.pt` (COCO weights) the first time you run
`YOLO("yolov8n.pt")` - useful to prove the pipeline runs end-to-end on day 1,
but remember it has never seen a license plate, so it won't detect any yet.

## 2. Get a plate dataset (Day 1-2)

Best starting point: **Roboflow Universe - "License Plate Recognition"**
(roboflow-universe-projects, ~24k images, YOLO-format labels, single class):
https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e

On the dataset page: Download Dataset → format **YOLOv8** → it gives you a
folder with `images/`, `labels/`, and a `data.yaml` already in the right
shape (see `plates.yaml.example` in this folder for what that looks like).
No account needed for the download link, a free Roboflow login is enough if
prompted.

If that one doesn't work for your demo footage's camera angle, a few
backups (also YOLOv8-ready, browse "Object Detection" on the page and grab
whichever preview images look closest to your traffic-camera angle):
- "Car License Plate" (public domain, smaller, quick to test with)
- "YOLOv8 number plate detection" (~5,750 images)

## 3. Run pretrained YOLO on sample images (Day 2)

```bash
python detect_plates.py sample.jpg yolov8n.pt
```

This won't find plates yet (COCO weights, no plate class) - it's just to
confirm ultralytics, OpenCV, and your install all work before you wait on a
dataset download.

## 4. Fine-tune on the plate dataset

Once you've got the dataset downloaded and `plates.yaml` pointing at it:

```bash
python train_plate_model.py --data plates.yaml --epochs 50
```

- Start with `yolov8n` (nano) - it's fast to train and plenty accurate for a
  single-class (plate) problem; only move to `yolov8s` if you have GPU time
  to spare and accuracy is lacking.
- 50 epochs is a reasonable default for a dataset this size; `patience=15`
  in the script stops early if it plateaus, so you won't waste your 14 days
  babysitting a training run.
- Best weights land at `runs/detect/train/weights/best.pt`. Point
  `detect_plates.load_model("runs/detect/train/weights/best.pt")` at that
  going forward instead of `yolov8n.pt`.
- No GPU on your laptop? Google Colab's free tier (T4 GPU) handles this
  fine for a dataset this size - a 50-epoch run on ~24k images typically
  finishes in under an hour on a T4.

## 5. Run on real traffic video (Day 4)

```python
from detect_plates import extract_frames, load_model, detect_plates

frames = extract_frames("traffic_sample.mp4", "frames/", every_n_seconds=1.5)
model = load_model("runs/detect/train/weights/best.pt")

for frame_path in frames:
    dets = detect_plates(frame_path, model)
    # hand dets off to your teammate's read_plate(), or to Member 4 for DB insert
```

## 6. Tuning tips

- If you're getting false positives on things that aren't plates, raise
  `conf_threshold` in `detect_plates()` (try 0.45-0.5).
- If you're missing plates that are small/far from the camera, lower it
  (0.2-0.25) and/or increase `imgsz` in training (e.g. 960 instead of 640) -
  costs more training time but helps with small objects.
- Keep a folder of `detections_preview.jpg`-style annotated outputs as you
  go - useful both for your own debugging and as screenshots for Member 6's
  PPT ("ANPR approach" slide, Day 4).

## Files in this folder

| File | Purpose |
|---|---|
| `detect_plates.py` | Core module: `detect_plates()`, `draw_detections()`, `extract_frames()` |
| `train_plate_model.py` | Fine-tunes YOLOv8 on your plate dataset |
| `plates.yaml.example` | What your dataset's `data.yaml` should look like |
| `requirements.txt` | pip dependencies |
