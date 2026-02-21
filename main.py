import cv2 as cv
import numpy as np
from pathlib import Path

from src.window import Window
from src.camera import Camera
from src.drone_detection import DroneDetector

PATH = Path("drone-tracking-datasets")

def main():
    window = Window()
    camera = Camera(PATH / "dataset1" / "cam0.mp4")
    camera.load_from_file()
    detector = DroneDetector(model_path=Path("models/yolo11s.pt"), model_size="s", use_sahi=True, use_motion_roi=False, slice_height=896, slice_width=896, imgsz=128, half=True)
    
    while True:
        frame = camera.get_frame()
        
        if frame is None:
            break
        
        detections = detector.detect(frame)
        
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            
            cv.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            cv.putText(frame, f"{confidence:.2f}", pt1, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        window.show_image(frame)
        cv.pollKey()
        
    window.close()
    camera.release()

if __name__ == "__main__":
    main()