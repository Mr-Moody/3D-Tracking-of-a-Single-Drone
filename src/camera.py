import numpy as np
import cv2 as cv
from pathlib import Path

class Camera():
    def __init__(self, filepath:Path=None):
        self.filepath = filepath

        self.capture = None
        self.current_frame = 0

    def load_from_file(self):
        self.capture = cv.VideoCapture(self.filepath)
        
        if not self.capture.isOpened():
            raise ValueError("Could not open video file")
        
        self.current_frame = 0
        
    def get_frame(self):
        """Read next frame from video. Only one frame in memory at a time."""
        ret, frame = self.capture.read()
        
        if not ret:
            return None  # end of video or read error
        
        self.current_frame += 1
        
        return frame

    def release(self):
        """Release the video stream and reset the current frame to 0"""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self.current_frame = 0