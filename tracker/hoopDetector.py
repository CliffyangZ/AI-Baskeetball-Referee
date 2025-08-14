"""
Basketball Hoop Detector with OpenVINO Runtime

This module implements basketball hoop detection using OpenVINO for
optimized inference, designed to work with the AI referee system.
Focuses specifically on hoop detection, not basketball detection.
"""

import numpy as np
import cv2
import openvino as ov
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import utilities
from tracker.utils.drawing import draw_hoop_detections, draw_frame_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported OpenVINO device types"""
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    AUTO = "AUTO"


@dataclass
class HoopDetection:
    """Detection result for basketball hoop"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[float, float] = None  # center coordinates (x, y)
    rim_center: Tuple[float, float] = None  # estimated rim center
    
    def __post_init__(self):
        """Calculate center and rim center from bounding box"""
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        # Rim is typically at the bottom center of the hoop detection
        self.rim_center = ((x1 + x2) / 2, y2 - (y2 - y1) * 0.15)


class HoopDetector:
    """
    Basketball Hoop Detector using OpenVINO Runtime
    
    Features:
    - OpenVINO optimized inference
    - Hoop coordinate extraction for AI referee
    - Configurable confidence threshold
    - Rim center estimation
    - Focus on hoop detection only
    """
    
    def __init__(
        self,
        model_path: str,
        device: DeviceType = DeviceType.CPU,
        confidence_thresh: float = 0.3,
        nms_thresh: float = 0.4
    ):
        """
        Initialize Hoop Detector
        
        Args:
            model_path: Path to OpenVINO IR model (.xml file)
            device: OpenVINO device type
            confidence_thresh: Confidence threshold for hoop detection
            nms_thresh: Non-maximum suppression threshold
        """
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        
        # Initialize OpenVINO
        self._init_openvino()
        
        # Detection state
        self.frame_count = 0
        self.last_hoops: List[HoopDetection] = []
        
        logger.info(f"HoopDetector initialized with device: {device.value}")
        logger.info(f"Confidence threshold: {confidence_thresh}")
    
    def _init_openvino(self):
        """Initialize OpenVINO Core and compile model"""
        try:
            # Initialize OpenVINO Core
            self.core = ov.Core()
            
            # Read model
            logger.info(f"Loading hoop model from: {self.model_path}")
            self.model = self.core.read_model(self.model_path)
            
            # Compile model
            self.compiled_model = self.core.compile_model(
                model=self.model, 
                device_name=self.device.value
            )
            
            # Get input/output info
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            
            # Get input shape
            self.input_shape = self.input_layer.shape
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
            
            logger.info(f"Model input shape: {self.input_shape}")
            logger.info(f"Model compiled successfully on {self.device.value}")
            
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
    
    def postprocess_detections(
        self, 
        outputs: np.ndarray, 
        frame_shape: Tuple[int, int]
    ) -> List[HoopDetection]:
        """
        Postprocess model outputs to extract hoop detections
        
        Args:
            outputs: Raw model outputs
            frame_shape: Original frame shape (height, width)
            
        Returns:
            List of hoop detections
        """
        detections = []
        frame_height, frame_width = frame_shape
        
        # Handle different output formats
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # Remove batch dimension
        
        # Check if we need to transpose (YOLOv8 format might be [classes+4, detections])
        if outputs.shape[0] < outputs.shape[1] and outputs.shape[0] <= 85:
            outputs = outputs.T
        
        # Process each detection
        for detection in outputs:
            if len(detection) >= 6:
                # Model format: [x_center, y_center, width, height, class1_conf, class2_conf]
                x_center, y_center, width, height = detection[:4]
                class1_conf = detection[4]  # basketball confidence
                class2_conf = detection[5]  # hoop confidence
                
                # Use hoop confidence
                confidence = class2_conf
                
                # Filter by confidence threshold
                if confidence >= self.confidence_thresh:
                    # Scale coordinates to frame size
                    scale_x = frame_width / self.input_width
                    scale_y = frame_height / self.input_height
                    
                    x_center_px = x_center * scale_x
                    y_center_px = y_center * scale_y
                    width_px = width * scale_x
                    height_px = height * scale_y
                    
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
                    
                    # Create hoop detection
                    hoop_detection = HoopDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(confidence)
                    )
                    detections.append(hoop_detection)
        
        # Apply Non-Maximum Suppression
        if detections:
            detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[HoopDetection]) -> List[HoopDetection]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) <= 1:
            return detections
        
        # Extract bounding boxes and scores
        boxes = np.array([det.bbox for det in detections])
        scores = np.array([det.confidence for det in detections])
        
        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.confidence_thresh, 
            self.nms_thresh
        )
        
        # Return filtered detections
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []
    
    def infer_frame(self, frame: np.ndarray) -> Tuple[List[HoopDetection], np.ndarray]:
        """
        Run inference on a single frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (hoop_detections, annotated_frame)
        """
        self.frame_count += 1
        
        try:
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            # Run inference
            results = self.compiled_model([input_tensor])
            outputs = results[self.output_layer]
            
            # Postprocess detections
            detections = self.postprocess_detections(outputs, frame.shape[:2])
            
            # Update last known hoops
            self.last_hoops = detections
            
            # Draw detections on frame
            annotated_frame = self.draw_hoops(frame.copy(), detections)
            
            return detections, annotated_frame
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return [], frame.copy()
    
    def draw_hoops(self, frame: np.ndarray, detections: List[HoopDetection]) -> np.ndarray:
        """Draw hoop detections on frame using drawing utilities"""
        return draw_hoop_detections(frame, detections)
    
    def get_hoop_coordinates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get hoop coordinates for AI referee system
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of dictionaries containing hoop information
        """
        detections, _ = self.infer_frame(frame)
        
        hoop_coords = []
        for i, detection in enumerate(detections):
            hoop_info = {
                'hoop_id': i + 1,
                'bbox': detection.bbox,
                'center': detection.center,
                'rim_center': detection.rim_center,
                'confidence': detection.confidence,
                'frame_number': self.frame_count
            }
            hoop_coords.append(hoop_info)
        
        return hoop_coords
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get OpenVINO device information"""
        try:
            available_devices = self.core.available_devices
            device_info = {
                "available_devices": available_devices,
                "current_device": self.device.value,
                "model_path": str(self.model_path),
                "input_shape": self.input_shape,
                "confidence_threshold": self.confidence_thresh
            }
            return device_info
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}


# Optimized Hoop Model class for compatibility with existing UI code
class OptimizedHoopModel:
    """Wrapper class for compatibility with existing UI code"""
    
    def __init__(self, model_path: str, device: str = "CPU"):
        device_enum = DeviceType(device.upper())
        self.detector = HoopDetector(model_path, device_enum)
    
    def infer_frame(self, frame: np.ndarray) -> np.ndarray:
        """Infer frame and return annotated result"""
        _, annotated_frame = self.detector.infer_frame(frame)
        return annotated_frame
    
    def get_hoop_coordinates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Get hoop coordinates for AI referee"""
        return self.detector.get_hoop_coordinates(frame)


if __name__ == "__main__":
    # Example usage
    model_path = "models/ov_models/hoopModel_openvino_model/hoopModel.xml"
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Please ensure the hoop detection model is available")
        exit(1)
    
    # Initialize detector
    detector = HoopDetector(model_path, DeviceType.CPU)
    
    # Process video
    video_path = "./data/video/parallel_angle.mov"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        exit(1)
    
    logger.info("Hoop Detection Demo")
    logger.info("Press 'q' to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        # Display results
        detections, annotated_frame = detector.infer_frame(frame)
        
        # Print hoop coordinates every 30 frames to reduce output
        if frame_count % 30 == 0:
            logger.info(f"Frame {frame_count}: Found {len(detections)} hoops")
            for i, detection in enumerate(detections):
                print(f"Hoop {i+1}: Rim at {detection.rim_center}, Confidence: {detection.confidence:.2f}")
        
        # Add frame info and show
        if annotated_frame is not None and annotated_frame.size > 0:
            annotated_frame = draw_frame_info(annotated_frame, frame_count, len(detections))
            cv2.imshow("Hoop Detection", annotated_frame)
        else:
            cv2.imshow("Hoop Detection", frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Quit requested")
            break
        elif key == ord('s'):
            # Save current frame
            save_path = f"hoop_detection_frame_{frame_count}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            logger.info(f"Saved frame to {save_path}")
    
    cap.release()
    cv2.destroyAllWindows()