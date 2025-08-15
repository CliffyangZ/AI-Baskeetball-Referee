"""
utils.py
------------
Utility functions for the AI Basketball Referee app.
Includes DPI awareness setup, model loading, frame preprocessing, and more.
"""

ASPECT_RATIO = 16 / 9

import ctypes, flet as ft

def get_scaled_size(base_width: int, aspect_ratio=(16, 9)):
    """Return (width, height) adjusted for system DPI and given aspect ratio."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    dpi_x = user32.GetDpiForSystem()
    scale_factor = dpi_x / 96

    width = int(base_width * scale_factor)
    height = int(width * aspect_ratio[1] / aspect_ratio[0])
    return width, height

def get_windows_scale_factor():
    """Detect Windows display scaling (DPI) and return scale factor."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        user32.SetProcessDPIAware()
        dpi = gdi32.GetDeviceCaps(user32.GetDC(0), 88)  # 88 = LOGPIXELSX
        return dpi / 96  # 96 is default DPI (100%)
    except Exception:
        return 1.0  # Fallback to 100% if detection fails


def make_aspect_ratio_handler(page, aspect_ratio=ASPECT_RATIO):
    """
    Returns a window event handler that enforces a fixed aspect ratio.
    """
    def on_window_event(e: ft.ControlEvent):
        if e.name == "resize":
            width = e.page.window_width
            height = e.page.window_height
            current_ratio = width / height if height != 0 else 0

            if abs(current_ratio - aspect_ratio) > 0.01:
                # Adjust height based on current width
                new_height = int(width / aspect_ratio)
                e.page.window_height = new_height
                e.page.update()

    return on_window_event

def calculate_frame_rate(frame_count, elapsed_time):
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0

def load_model(model_path):
    # Load the OpenVINO model from the specified path
    from openvino.inference_engine import IECore
    ie = IECore()
    net = ie.read_network(model=model_path + '/model.xml', weights=model_path + '/model.bin')
    return ie, net

def preprocess_frame(frame):
    # Preprocess the frame for model inference
    from cv2 import resize
    return resize(frame, (640, 480))  # Resize to the model's input size

def postprocess_results(results):
    # Process the inference results and return meaningful output
    return results  # Placeholder for actual postprocessing logic

def draw_bounding_box(frame, box, label):
    # Draw a bounding box on the frame
    from cv2 import rectangle, putText
    x, y, w, h = box
    rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    putText(frame, label, (x, y - 10), 1, 1, (0, 255, 0), 2)