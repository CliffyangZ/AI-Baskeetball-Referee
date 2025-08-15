"""
OpenVINO utilities for basketball and pose tracking
Common functions and classes to reduce code duplication
"""

import numpy as np
import cv2
import openvino as ov
from typing import Tuple
from enum import Enum
import logging
from pathlib import Path
import time
import threading

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported OpenVINO device types"""
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    AUTO = "AUTO"


class OpenVINOInferenceEngine:
    """
    OpenVINO inference engine wrapper with thread synchronization
    """
    
    def __init__(self, model_path: str, device: str = "CPU"):
        """
        Initialize OpenVINO inference engine
        
        Args:
            model_path: Path to OpenVINO IR model (.xml)
            device: Target device (CPU, GPU, etc.)
        """
        self.model_path = model_path
        self.device = device
        
        # Initialize OpenVINO components
        self._init_openvino()
        
        logger.info(f"OpenVINO engine initialized with device: {device}")
    
    def _init_openvino(self):
        """Initialize OpenVINO Core and compile model"""
        try:
            # Initialize OpenVINO Core
            self.core = ov.Core()
            
            # Read model
            logger.info(f"Loading model from: {self.model_path}")
            self.model = self.core.read_model(self.model_path)
            
            # Compile model
            # Handle DeviceType enum or string
            device_str = self.device.value if hasattr(self.device, 'value') else str(self.device)
            self.compiled_model = self.core.compile_model(
                model=self.model, 
                device_name=device_str
            )
            
            # Get input/output info
            self.input_layer = self.compiled_model.inputs[0]
            
            # Use output index instead of name to avoid tensor name issues
            self.output_index = 0
            
            # Get input shape
            self.input_shape = self.input_layer.shape
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
            
            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
            
            # Initialize FPS counter
            self.fps_counter = FPSCounter()
            
            # Add thread lock for inference synchronization
            self.inference_lock = threading.Lock()
        
        except Exception as e:
            logger.error(f"Failed to initialize OpenVINO: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model inference
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Resize frame to model input size
        resized_frame = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized_frame = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW format
        input_tensor = np.transpose(normalized_frame, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def infer(self, input_tensor: np.ndarray, max_retries: int = 5, retry_delay: float = 0.05) -> np.ndarray:
        """
        Run inference on preprocessed tensor with thread synchronization and retry logic
        
        Args:
            input_tensor: Preprocessed input tensor
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Raw model outputs
        """
        retry_count = 0
        last_error = None
        backoff_factor = 1.5  # Exponential backoff factor
        current_delay = retry_delay
        
        # Try to acquire the lock with timeout
        lock_acquired = self.inference_lock.acquire(timeout=1.0)
        if not lock_acquired:
            logging.warning("Could not acquire inference lock, proceeding without lock")
        
        try:
            while retry_count < max_retries:
                try:
                    results = self.compiled_model([input_tensor])
                    # Use output index instead of name
                    return results[self.output_index]
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    if "Infer Request is busy" in str(e):
                        # If the model is busy, wait with exponential backoff
                        time.sleep(current_delay)
                        current_delay *= backoff_factor  # Increase delay for next retry
                        continue
                    else:
                        # For other errors, raise immediately
                        raise
            
            # If we've exhausted all retries, log and raise the last error
            logging.error(f"Inference failed after {max_retries} retries: {last_error}")
            return np.zeros((1, 1))  # Return empty result instead of crashing
        finally:
            # Always release the lock if we acquired it
            if lock_acquired:
                self.inference_lock.release()


class FPSCounter:
    """
    FPS calculation utility for performance monitoring
    """
    
    def __init__(self, update_interval: int = 30):
        """
        Initialize FPS counter
        
        Args:
            update_interval: Number of frames between FPS updates
        """
        self.update_interval = update_interval
        self.reset()
    
    def reset(self):
        """Reset FPS counter"""
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0
    
    def update(self) -> float:
        """
        Update FPS calculation
        
        Returns:
            Current FPS value
        """
        self.frame_count += 1
        if self.frame_count % self.update_interval == 0:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            self.current_fps = self.update_interval / elapsed_time
            self.start_time = current_time
        
        return self.current_fps
    
    def get_fps(self) -> float:
        """Get current FPS value"""
        return self.current_fps


def draw_fps_info(frame: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Draw FPS information on frame
    
    Args:
        frame: Input frame
        fps: FPS value to display
        position: Text position (x, y)
        
    Returns:
        Frame with FPS text
    """
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, position, 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame


def draw_detection_count(frame: np.ndarray, count: int, label: str = "Detections", 
                        position: Tuple[int, int] = (10, 70)) -> np.ndarray:
    """
    Draw detection count on frame
    
    Args:
        frame: Input frame
        count: Number of detections
        label: Label text
        position: Text position (x, y)
        
    Returns:
        Frame with count text
    """
    count_text = f"{label}: {count}"
    cv2.putText(frame, count_text, position, 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame


class BaseOptimizedModel:
    """
    Base class for optimized model wrappers
    Provides common interface for UI compatibility
    """
    
    def __init__(self, model_path: str, device: str = "CPU"):
        """
        Initialize base optimized model
        
        Args:
            model_path: Path to model file
            device: Device string (CPU/GPU/NPU/AUTO)
        """
        self.device_enum = DeviceType(device.upper())
        self.model_path = model_path
    
    def infer_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Infer frame and return annotated result
        Must be implemented by subclasses
        """
        raise NotImplementedError("Subclass must implement infer_frame")


def normalize_coordinates(coords: Tuple[float, float, float, float], 
                         input_size: Tuple[int, int], 
                         frame_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """
    Convert normalized coordinates to pixel coordinates
    
    Args:
        coords: Normalized coordinates (x_center, y_center, width, height)
        input_size: Model input size (width, height)
        frame_size: Frame size (width, height)
        
    Returns:
        Pixel coordinates (x1, y1, x2, y2)
    """
    x_center, y_center, width, height = coords
    input_width, input_height = input_size
    frame_width, frame_height = frame_size
    
    # Convert to pixel coordinates
    x_center_px = (x_center / input_width) * frame_width
    y_center_px = (y_center / input_height) * frame_height
    width_px = (width / input_width) * frame_width
    height_px = (height / input_height) * frame_height
    
    # Convert to corner coordinates
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    x2 = x_center_px + width_px / 2
    y2 = y_center_px + height_px / 2
    
    # Ensure coordinates are within frame bounds
    x1 = max(0, min(x1, frame_width))
    y1 = max(0, min(y1, frame_height))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))
    
    return (x1, y1, x2, y2)


def ensure_frame_bounds(coords: Tuple[float, float], frame_size: Tuple[int, int]) -> Tuple[float, float]:
    """
    Ensure coordinates are within frame bounds
    
    Args:
        coords: Coordinates (x, y)
        frame_size: Frame size (width, height)
        
    Returns:
        Bounded coordinates
    """
    x, y = coords
    frame_width, frame_height = frame_size
    
    x = max(0, min(x, frame_width))
    y = max(0, min(y, frame_height))
    
    return (x, y)


def draw_detection_info(frame: np.ndarray, detections, tracks, fps: float) -> np.ndarray:
    """
    Draw detection and tracking information on frame
    
    Args:
        frame: Input frame
        detections: List of detections
        tracks: List of tracks
        fps: Current FPS value
        
    Returns:
        Frame with information overlay
    """
    # Draw FPS
    frame = draw_fps_info(frame, fps)
    
    # Draw detection count
    frame = draw_detection_count(frame, len(detections), "Detections")
    
    # Draw track count
    track_text = f"Tracks: {len(tracks)}"
    cv2.putText(frame, track_text, (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return frame
