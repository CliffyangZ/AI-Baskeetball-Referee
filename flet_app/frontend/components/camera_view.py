from flet import Camera

class CameraView:
    def __init__(self, on_frame):
        self.camera = Camera(on_frame=on_frame)

    def start(self):
        self.camera.start()

    def stop(self):
        self.camera.stop()

    def get_view(self):
        return self.camera.view()