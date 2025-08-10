import base64
import cv2
import numpy as np


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_to_png_bytes(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.png', frame)
    if not ok:
        raise RuntimeError('Failed to encode frame to PNG')
    return buf.tobytes()


def frame_to_base64_png(frame: np.ndarray) -> str:
    return base64.b64encode(frame_to_png_bytes(frame)).decode('ascii')
