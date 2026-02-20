import cv2 as cv
import numpy as np

class Window():
    def __init__(self, title="Image"):
        self.title = title
        self.window = cv.namedWindow(title)
        self.image = None

    def show_image(self, image):
        self.image = image
        cv.imshow(self.title, image)

    def close(self):
        cv.destroyWindow(self.title)

    def draw_annotation(self, bbox):
        x, y, w, h = bbox
        cv.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
