"""
Basketball Tracker with OpenVINO Runtime Inference and ByteTrack Algorithm

This module implements basketball detection and tracking using OpenVINO for
optimized inference and ByteTrack for multi-object tracking.
Designed for AI referee system integration with modular architecture.
"""

import numpy as np
import cv2
import openvino as ov
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import time

# Import utilities
try:
    from .utils.byte_track import BYTETracker, STrack
    from .utils.KalmanFilter import BasketballKalmanFilter
    from .utils.drawing import draw_basketball_track, draw_detection_info
    from .utils.matching import calculate_iou
    from .utils.openvino_utils import (
        DeviceType, OpenVINOInferenceEngine, FPSCounter, 
        normalize_coordinates, ensure_frame_bounds, BaseOptimizedModel
    )
except ImportError:
    from utils.byte_track import BYTETracker, STrack
    from utils.KalmanFilter import BasketballKalmanFilter
    from utils.drawing import draw_basketball_track, draw_detection_info
    from utils.matching import calculate_iou
    from utils.openvino_utils import (
        DeviceType, OpenVINOInferenceEngine, FPSCounter, 
        normalize_coordinates, ensure_frame_bounds, BaseOptimizedModel
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# DeviceType now imported from openvino_utils

@dataclass
class BasketballDetection:
    """Basketball detection data structure"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    
    @property
    def center_x(self) -> float:
        """Calculate center x coordinate"""
        return (self.bbox[0] + self.bbox[2]) / 2
    
    @property
    def center_y(self) -> float:
        """Calculate center y coordinate"""
        return (self.bbox[1] + self.bbox[3]) / 2
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center coordinates (required by ByteTrack)"""
        return (self.center_x, self.center_y)
    
    @property
    def width(self) -> float:
        """Calculate bounding box width"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Calculate bounding box height"""
        return self.bbox[3] - self.bbox[1]


class BasketballTracker:
    """
    Basketball Tracker using OpenVINO Runtime and ByteTrack algorithm
    Modular design with separated inference, tracking, and visualization
    
    Features:
    - OpenVINO optimized inference
    - ByteTrack multi-object tracking algorithm
    - Modular utilities for drawing and matching
    - Basketball coordinate output for AI referee
    - Real-time performance monitoring
    """
    
    def __init__(
        self,
        model_path: str,
        device: DeviceType = DeviceType.CPU,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30
    ):
        """
        Initialize Basketball Tracker for AI referee
        
        Args:
            model_path: Path to OpenVINO IR model (.xml file)
            device: OpenVINO device type
            high_thresh: High confidence threshold for ByteTrack first association
            low_thresh: Low confidence threshold for ByteTrack second association  
            match_thresh: IoU threshold for track association
            track_buffer: Number of frames to keep lost tracks
            frame_rate: Video frame rate for timing calculations
        """
        self.model_path = Path(model_path)
        self.device = device
        
        # Initialize OpenVINO engine
        self.inference_engine = OpenVINOInferenceEngine(model_path, device)
        
        # Initialize ByteTracker
        self.tracker = BYTETracker(
            high_thresh=high_thresh,
            low_thresh=low_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate
        )
        
        # Performance monitoring
        self.fps_counter = FPSCounter()
        
        logger.info(f"BasketballTracker initialized with device: {device.value}")
    
    # OpenVINO initialization and preprocessing now handled by inference_engine
    
    def postprocess_detections(
        self, 
        outputs: np.ndarray, 
        frame_shape: Tuple[int, int]
    ) -> List[BasketballDetection]:
        """
        Postprocess model outputs to extract basketball detections
        
        Args:
            outputs: Raw model outputs
            frame_shape: Original frame shape (height, width)
            
        Returns:
            List of basketball detections
        """
        detections = []
        frame_height, frame_width = frame_shape
        
        # Handle different output formats
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # Remove batch dimension
        
        # Check if we need to transpose (YOLOv8 format might be [85, 8400])
        if outputs.shape[0] < outputs.shape[1] and outputs.shape[0] <= 85:
            outputs = outputs.T
        
        # Process each detection
        for detection in outputs:
            if len(detection) >= 5:
                # YOLOv8 format: [x_center, y_center, width, height, confidence, ...]
                x_center, y_center, width, height, confidence = detection[:5]
                
                # Filter by low threshold (ByteTrack will handle high/low separation)
                if confidence >= self.tracker.low_thresh:
                    # Convert normalized coordinates to pixel coordinates using utils
                    input_size = (self.inference_engine.input_width, self.inference_engine.input_height)
                    frame_size = (frame_width, frame_height)
                    x1, y1, x2, y2 = normalize_coordinates(
                        (x_center, y_center, width, height), input_size, frame_size
                    )
                    
                    # Create detection
                    basketball_detection = BasketballDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(confidence)
                    )
                    detections.append(basketball_detection)
        
        # Apply Non-Maximum Suppression to remove duplicate detections
        detections = self.apply_nms(detections, iou_threshold=0.5)
        
        return detections
    
    def apply_nms(self, detections: List[BasketballDetection], iou_threshold: float = 0.5) -> List[BasketballDetection]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) <= 1:
            return detections
        
        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply NMS
        keep_detections = []
        while detections:
            # Keep the detection with highest confidence
            best_detection = detections.pop(0)
            keep_detections.append(best_detection)
            
            # Remove detections with high IoU overlap
            remaining_detections = []
            for detection in detections:
                iou = calculate_iou(best_detection.bbox, detection.bbox)
                if iou <= iou_threshold:
                    remaining_detections.append(detection)
            
            detections = remaining_detections
        
        return keep_detections
    
    def infer_frame(self, frame: np.ndarray) -> Tuple[List[STrack], np.ndarray]:
        """
        Run inference on a single frame for AI referee
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (active_tracks, annotated_frame)
        """
        try:
            # Update FPS counter
            current_fps = self.fps_counter.update()
            
            # Preprocess frame and run inference using engine
            input_tensor = self.inference_engine.preprocess_frame(frame)
            outputs = self.inference_engine.infer(input_tensor)
            
            # Postprocess detections
            detections = self.postprocess_detections(outputs, frame.shape[:2])
            
            # Update tracks using ByteTracker
            active_tracks = self.tracker.update(detections)
            
            # Draw annotations
            annotated_frame = self.draw_tracks(frame.copy(), active_tracks, detections)
            
            return active_tracks, annotated_frame
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return [], frame.copy()
    
    def get_basketball_coordinates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get basketball coordinates for AI referee system
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of dictionaries containing basketball information
        """
        tracks, _ = self.infer_frame(frame)
        return self.tracker.get_basketball_coordinates()
    
    def draw_tracks(self, 
                   frame: np.ndarray, 
                   tracks: List[STrack], 
                   detections: List[BasketballDetection]) -> np.ndarray:
        """
        Draw basketball tracks and detection info on frame
        
        Args:
            frame: Input frame
            tracks: List of active tracks
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        # Draw each track
        for track in tracks:
            frame = draw_basketball_track(
                frame, track, 
                show_trajectory=True, 
                show_velocity=True
            )
        
        # Draw detection statistics
        frame = draw_detection_info(
            frame, detections, tracks, self.fps_counter.get_fps()
        )
        
        return frame
    
    def reset(self):
        """Reset tracker state"""
        self.tracker.reset()
        self.fps_counter.reset()


# Optimized Basketball Model class for compatibility with existing UI code
class OptimizedBasketballModel(BaseOptimizedModel):
    """Wrapper class for compatibility with existing UI code"""
    
    def __init__(self, model_path: str, device: str = "CPU"):
        super().__init__(model_path, device)
        self.tracker = BasketballTracker(model_path, self.device_enum)
    
    def infer_frame(self, frame: np.ndarray) -> np.ndarray:
        """Infer frame and return annotated result"""
        _, annotated_frame = self.tracker.infer_frame(frame)
        return annotated_frame
    
    def get_basketball_coordinates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Get basketball coordinates for AI referee"""
        return self.tracker.get_basketball_coordinates(frame)


if __name__ == "__main__":
    # Example usage for AI referee
    model_path = "models/ov_models/basketballModel_openvino_model/basketballModel.xml"
    tracker = BasketballTracker(model_path, DeviceType.CPU)
    
    # Process video
    cap = cv2.VideoCapture("./data/video/dribbling.mov")
    
    print("Basketball Tracker with ByteTrack Algorithm")
    print("Press 'q' to quit, 'r' to reset tracker")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Get basketball coordinates for AI referee
        ball_coords = tracker.get_basketball_coordinates(frame)
        
        # Display results
        tracks, annotated_frame = tracker.infer_frame(frame)
        
        # Print basketball coordinates for AI referee
        for ball in ball_coords:
            print(f"Ball {ball['ball_id']}: Center at ({ball['center'][0]:.1f}, {ball['center'][1]:.1f}), "
                  f"Velocity: ({ball['velocity'][0]:.1f}, {ball['velocity'][1]:.1f})")
        
        cv2.imshow("Basketball Tracking with ByteTrack", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset()
            print("Tracker reset")
    
    cap.release()
    cv2.destroyAllWindows()