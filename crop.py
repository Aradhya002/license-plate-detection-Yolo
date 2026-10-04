import cv2
from detect_plates import load_model, detect_plates

model = load_model("best.pt")
dets = detect_plates("sample.jpeg", model)

for i, d in enumerate(dets):
    cv2.imwrite(f"plate_only_{i}.jpg", d["crop"])