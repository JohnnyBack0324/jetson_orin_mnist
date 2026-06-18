#!/usr/bin/env python3
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="dataset/data.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    workers=0,          # ★ shm 안 씀 (No space left on device 회피)
    patience=20,
    name="digit_detector"
)
print("done -> runs/detect/digit_detector/weights/best.pt")
