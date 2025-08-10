"""
Basketball Hoop Detector with OpenVINO Runtime Inference

This module implements basketball hoop detection using OpenVINO for
optimized inference, designed to work with the AI referee system.
"""

import numpy as np
import cv2
import openvino as ov
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

# Import utilities
from utils.image_utils import bgr_to_rgb, frame_to_base64_png

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported OpenVINO device types"""
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    AUTO = "AUTO"


@dataclass
class HoopDetection:
    """Detection result for basketball objects (hoop or basketball)"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0  # 0=basketball, 1=hoop (or vice versa)
    class_name: str = "unknown"  # "basketball" or "hoop"
    center: Tuple[float, float] = None  # center coordinates (x, y)
    rim_center: Tuple[float, float] = None  # estimated rim center
    
    def __post_init__(self):
        """Calculate center and rim center from bounding box"""
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        # Rim is typically at the bottom center of the hoop detection
        self.rim_center = ((x1 + x2) / 2, y2 - (y2 - y1) * 0.1)


class HoopDetector:
    """
    Basketball Hoop Detector using OpenVINO Runtime
    
    Features:
    - OpenVINO optimized inference
    - Hoop coordinate extraction for AI referee
    - Configurable confidence threshold
    - Rim center estimation
    """
    
    def __init__(
        self,
        model_path: str,
        device: DeviceType,
        confidence_thresh: float = 0.1,
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
        
        # Debug: Print output shape and sample values
        logger.debug(f"Raw output shape: {outputs.shape}")
        logger.debug(f"Frame shape: {frame_shape}")
        
        # Handle different output formats
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # Remove batch dimension
            logger.debug(f"After removing batch dim: {outputs.shape}")
        
        # For YOLO models, outputs might be in format [num_detections, 85] where 85 = 4(bbox) + 1(conf) + 80(classes)
        # Or it could be [8400, 85] for YOLOv8 format
        
        # Scale factors for converting back to original frame size
        scale_x = frame_width / self.input_width
        scale_y = frame_height / self.input_height
        
        logger.debug(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
        
        # Check if we need to transpose (YOLOv8 format might be [85, 8400])
        if outputs.shape[0] < outputs.shape[1] and outputs.shape[0] <= 85:
            outputs = outputs.T
            logger.debug(f"Transposed output shape: {outputs.shape}")
        
        detection_count = 0
        # Process each detection
        for i, detection in enumerate(outputs):
            if len(detection) >= 6:
                # For 6-element output: [x_center, y_center, width, height, class1_conf, class2_conf]
                # where class1 might be basketball and class2 might be hoop (or vice versa)
                x_center, y_center, width, height = detection[:4]
                
                # Get class confidences
                class1_conf = detection[4]  # Could be basketball confidence
                class2_conf = detection[5]  # Could be hoop confidence
                
                # Determine which class has higher confidence
                if class1_conf > class2_conf:
                    confidence = class1_conf
                    class_id = 0
                    class_name = "basketball"  # Assuming class 0 is basketball
                else:
                    confidence = class2_conf
                    class_id = 1
                    class_name = "hoop"  # Assuming class 1 is hoop
                
                # Debug first few detections
                if i < 3:
                    logger.debug(f"Detection {i}: center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")
                    logger.debug(f"  Class1 (basketball) conf: {class1_conf:.3f}, Class2 (hoop) conf: {class2_conf:.3f}")
                    logger.debug(f"  Predicted: {class_name} (class_id: {class_id}) with confidence: {confidence:.3f}")
                
                # Filter by confidence
                if confidence >= self.confidence_thresh:
                    detection_count += 1
                    
                    # The coordinates seem to be in a different format
                    # Let's try treating them as already in pixel coordinates relative to input size
                    x_center_px = (x_center / self.input_width) * frame_width
                    y_center_px = (y_center / self.input_height) * frame_height
                    width_px = (width / self.input_width) * frame_width
                    height_px = (height / self.input_height) * frame_height
                    
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
                    
                    logger.debug(f"Valid detection: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), class={class_name}, conf={confidence:.3f}")
                    
                    # Create detection with class information
                    hoop_detection = HoopDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(confidence),
                        class_id=class_id,
                        class_name=class_name
                    )
                    detections.append(hoop_detection)
        
        logger.debug(f"Found {detection_count} detections above threshold, returning {len(detections)} after processing")
        
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
    
    def calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
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
            
            logger.debug(f"Frame {self.frame_count}: Found {len(detections)} hoops")
            
            return detections, annotated_frame
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return [], frame.copy()
    
    def draw_hoops(self, frame: np.ndarray, detections: List[HoopDetection]) -> np.ndarray:
        """Draw detections on frame with correct class labels"""
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Choose colors based on class
            if detection.class_name == "basketball":
                box_color = (0, 255, 0)  # Green for basketball
                center_color = (0, 255, 0)
                coord_color = (0, 255, 255)  # Yellow for ball center
            elif detection.class_name == "hoop":
                box_color = (255, 0, 0)  # Blue for hoop
                center_color = (255, 0, 0)
                coord_color = (0, 0, 255)  # Red for rim
            else:
                box_color = (128, 128, 128)  # Gray for unknown
                center_color = (128, 128, 128)
                coord_color = (128, 128, 128)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw center point
            center_x, center_y = map(int, detection.center)
            cv2.circle(frame, (center_x, center_y), 5, center_color, -1)
            
            # Draw rim/ball center
            rim_x, rim_y = map(int, detection.rim_center)
            cv2.circle(frame, (rim_x, rim_y), 3, coord_color, -1)
            
            # Draw labels with correct class name
            class_display = detection.class_name.capitalize()
            label = f"{class_display} {i+1}: {detection.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Draw coordinate info based on class
            if detection.class_name == "basketball":
                coord_label = f"Ball: ({center_x}, {center_y})"
            else:
                coord_label = f"Rim: ({rim_x}, {rim_y})"
            cv2.putText(frame, coord_label, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, coord_color, 1)
        
        return frame
    
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
    detector = HoopDetector(model_path, DeviceType.CPU)
    
    # Process video
    cap = cv2.VideoCapture("./data/video/parallel_angle.mov")  # or video file path
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get hoop coordinates for AI referee
        hoop_coords = detector.get_hoop_coordinates(frame)
        
        # Display results
        detections, annotated_frame = detector.infer_frame(frame)
        
        # Print hoop coordinates
        for hoop in hoop_coords:
            print(f"Hoop {hoop['hoop_id']}: Rim at {hoop['rim_center']}, Confidence: {hoop['confidence']:.2f}")
        
        cv2.imshow("Hoop Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
