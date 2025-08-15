import cv2

class VideoCapture:
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise Exception("Could not open video device")

    def get_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            raise Exception("Could not read frame from video device")
        return frame

    def release(self):
        self.capture.release()

    def __del__(self):
        self.release()