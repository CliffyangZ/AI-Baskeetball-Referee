import cv2
import numpy as np
from collections import deque


def color_by_id(track_id: int) -> tuple:
    """Generate a deterministic bright BGR color based on an integer id."""
    seed = int(track_id * 9999) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    r, g, b = rng.integers(0, 256, size=3).tolist()
    if (r + g + b) < 300:
        r = min(255, r + 100)
        g = min(255, g + 100)
        b = min(255, b + 100)
    return int(b), int(g), int(r)  # BGR for OpenCV


def draw_bbox(frame, x1, y1, x2, y2, color, thickness: int = 2):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def put_label(frame, text: str, x: int, y: int, color, scale: float = 0.5, thickness: int = 2):
    cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_center(frame, x: int, y: int, color, radius: int = 4):
    cv2.circle(frame, (int(x), int(y)), int(radius), color, -1)


def draw_trajectory(frame, trajectory: deque, color, thickness: int = 2):
    if len(trajectory) < 2:
        return
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], color, thickness)
