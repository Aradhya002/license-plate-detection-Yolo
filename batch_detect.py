"""
Run plate detection on every image inside a folder - no need to type
filenames one at a time anymore.

Usage:
    python batch_detect.py
        (looks for a folder called "input_images" by default)

    python batch_detect.py some_other_folder
        (uses that folder instead)

Output:
    output_crops/     - just the cropped plate, one file per plate found
    output_previews/  - the full original image with boxes drawn, one per input image
                         (saved even for images with 0 detections, so you can see what it missed)
"""

import sys
import cv2
from pathlib import Path
from detect_plates import load_model, detect_plates, draw_detections

INPUT_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "input_images"
MODEL_PATH = "best.pt"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    input_dir = Path(INPUT_FOLDER)
    if not input_dir.exists():
        print(f"Folder not found: {input_dir}")
        print(f"Create a folder called '{INPUT_FOLDER}' and put your images in it.")
        return

    crops_dir = Path("output_crops")
    previews_dir = Path("output_previews")
    crops_dir.mkdir(exist_ok=True)
    previews_dir.mkdir(exist_ok=True)

    model = load_model(MODEL_PATH)

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    print(f"Found {len(image_paths)} image(s) in '{input_dir}'\n")

    total_plates = 0
    for img_path in image_paths:
        dets = detect_plates(img_path, model)

        # Save a boxed preview either way - useful to see what got missed too
        draw_detections(img_path, dets, out_path=previews_dir / f"{img_path.stem}_preview.jpg")

        if not dets:
            print(f"  {img_path.name}: no plates found")
            continue

        for i, d in enumerate(dets):
            crop_path = crops_dir / f"{img_path.stem}_plate{i}.jpg"
            cv2.imwrite(str(crop_path), d["crop"])
            total_plates += 1

        print(f"  {img_path.name}: {len(dets)} plate(s) found")

    print(f"\nDone. {total_plates} plate detection(s) across {len(image_paths)} image(s).")
    print(f"Crops saved to    -> {crops_dir}/")
    print(f"Previews saved to -> {previews_dir}/")


if __name__ == "__main__":
    main()
