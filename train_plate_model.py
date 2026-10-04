"""
Fine-tune YOLOv8 on a license-plate dataset.

Only run this once you have a dataset in YOLO format (images/ + labels/ +
a data.yaml - see plates.yaml.example in this folder). Point --data at your
own data.yaml.

Usage:
    python train_plate_model.py --data plates.yaml --epochs 50
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--base", default="yolov8n.pt", help="base weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=15,  # stop early if val loss stalls - useful with a 14-day clock
    )
    # Best weights land at runs/detect/train/weights/best.pt -
    # that's the path to feed into detect_plates.load_model() afterwards.


if __name__ == "__main__":
    main()
