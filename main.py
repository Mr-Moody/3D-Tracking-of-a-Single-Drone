import cv2 as cv
import numpy as np
from pathlib import Path
import time

import torch

from src.window import Window
from src.camera import Camera
from src.drone_detection import DroneDetector

PATH = Path("drone-tracking-datasets")


def _verify_gpu() -> None:
    """Print GPU status and verify CUDA is available."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"GPU: {name} (CUDA available)")
    else:
        print("WARNING: GPU not available, using CPU")


def main():
    _verify_gpu()

    window = Window()
    camera = Camera(PATH / "dataset1" / "cam0.mp4")
    camera.load_from_file()
    detector = DroneDetector(model_path=Path("models/yolo11s.pt"), model_size="s", use_sahi=False, use_motion_roi=True, slice_height=896, slice_width=896, imgsz=320, half=True)

    cached_detections: list[tuple[float, float, float, float, float]] = []

    while True:
        start_time = time.time()
        frame = camera.get_frame()

        if frame is None:
            break

        detections = detector.detect(frame)
        
        if detections:
            cached_detections = detections
        else:
            detections = cached_detections

        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))

            cv.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            cv.putText(frame, f"{confidence:.2f}", pt1, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        window.show_image(frame)
        cv.pollKey()
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps}")
        
    window.close()
    camera.release()

if __name__ == "__main__":
    main()