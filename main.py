import cv2 as cv
import numpy as np
from pathlib import Path

from src.window import Window
from src.camera import Camera

PATH = Path("drone-tracking-datasets")

def main():
    window = Window()
    camera = Camera(PATH / "dataset1" / "cam0.mp4")
    camera.load_from_file()
    
    while True:
        frame = camera.get_frame()
        
        if frame is None:
            break
        
        window.show_image(frame)
        cv.waitKey(1000)
        
    window.close()
    camera.release()

if __name__ == "__main__":
    main()