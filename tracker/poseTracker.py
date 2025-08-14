"""
Pose Tracker with OpenVINO Runtime Inference for YOLO Pose Models

This module implements human pose detection using OpenVINO for optimized inference.
Designed for AI referee system integration with YOLO pose model support.
No tracking functionality - pure pose detection and keypoint extraction.
"""

import numpy as np
import cv2
import openvino as ov
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
import logging
from pathlib import Path
import time

# Import utilities
# Import directly from project root utils
from utils.image_utils import bgr_to_rgb, frame_to_base64_png
from utils.matching import calculate_iou
from utils.openvino_utils import (
    DeviceType, OpenVINOInferenceEngine, FPSCounter, 
    normalize_coordinates, ensure_frame_bounds, BaseOptimizedModel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# DeviceType now imported from openvino_utils


class PoseModel(Enum):
    """Supported pose model types"""
    YOLOV8_POSE = "yolov8_pose"
    YOLOV11_POSE = "yolov11_pose"


@dataclass
class Keypoint:
    """Single keypoint with coordinates and confidence"""
    x: float
    y: float
    confidence: float
    visible: bool = True
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple format"""
        return (self.x, self.y, self.confidence)


@dataclass
class PoseDetection:
    """Human pose detection result"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: List[Keypoint]  # 17 COCO keypoints
    confidence: float
    person_id: int = 0
    
    def __post_init__(self):
        """Calculate additional properties"""
        # Calculate center from bbox
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Calculate average keypoint confidence
        valid_keypoints = [kp for kp in self.keypoints if kp.confidence > 0.3]
        self.avg_keypoint_confidence = np.mean([kp.confidence for kp in valid_keypoints]) if valid_keypoints else 0.0


class PoseTracker:
    """
    Pose Tracker using OpenVINO Runtime for YOLO Pose Models
    
    Features:
    - OpenVINO optimized inference
    - YOLO pose model support (YOLOv8-Pose, YOLOv11-Pose)
    - COCO 17-keypoint format
    - Real-time pose detection
    - No tracking - pure detection
    """
    
    # COCO 17 keypoint names
    COCO_KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # COCO skeleton connections for drawing
    COCO_SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    def __init__(
        self,
        model_path: str,
        device: DeviceType = DeviceType.CPU,
        pose_model: PoseModel = PoseModel.YOLOV8_POSE,
        confidence_threshold: float = 0.5,
        keypoint_threshold: float = 0.3
    ):
        """
        Initialize Pose Tracker for AI referee
        
        Args:
            model_path: Path to OpenVINO IR model (.xml file)
            device: OpenVINO device type
            pose_model: Type of pose model
            confidence_threshold: Minimum confidence for pose detection
            keypoint_threshold: Minimum confidence for keypoint visibility
        """
        self.model_path = Path(model_path)
        self.device = device
        self.pose_model = pose_model
        self.confidence_threshold = confidence_threshold
        self.keypoint_threshold = keypoint_threshold
        
        # Initialize OpenVINO engine
        self.inference_engine = OpenVINOInferenceEngine(model_path, device)
        
        # Performance monitoring
        self.fps_counter = FPSCounter()
        
        logger.info(f"PoseTracker initialized with device: {device.value}, model: {pose_model.value}")
    
    # OpenVINO initialization and preprocessing now handled by inference_engine
    
    def apply_nms(self, detections: List[PoseDetection], iou_threshold: float = 0.5) -> List[PoseDetection]:
        """Apply Non-Maximum Suppression to remove duplicate detections using utils"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            # Keep the highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove detections with high IoU overlap using utils function
            remaining = []
            for det in detections:
                iou = calculate_iou(current.bbox, det.bbox)
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep

    def postprocess_detections(
        self, 
        outputs: np.ndarray, 
        frame_shape: Tuple[int, int]
    ) -> List[PoseDetection]:
        """
        Postprocess model outputs to extract pose detections
        
        Args:
            outputs: Raw model outputs
            frame_shape: Original frame shape (height, width)
            
        Returns:
            List of pose detections
        """
        detections = []
        frame_height, frame_width = frame_shape
        
        # Handle different output formats
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # Remove batch dimension
        
        # Check if we need to transpose (YOLOv8-Pose format might be [56, 8400])
        # Format: [x, y, w, h, conf, kp1_x, kp1_y, kp1_conf, ..., kp17_x, kp17_y, kp17_conf]
        if outputs.shape[0] < outputs.shape[1] and outputs.shape[0] <= 56:
            outputs = outputs.T
        
        # Process each detection
        for detection in outputs:
            if len(detection) >= 56:  # 4 bbox + 1 conf + 51 keypoints (17*3)
                # Extract bbox and confidence
                x_center, y_center, width, height, confidence = detection[:5]
                
                # Filter by confidence threshold
                if confidence >= self.confidence_threshold:
                    # Convert normalized coordinates to pixel coordinates using utils
                    input_size = (self.inference_engine.input_width, self.inference_engine.input_height)
                    frame_size = (frame_width, frame_height)
                    x1, y1, x2, y2 = normalize_coordinates(
                        (x_center, y_center, width, height), input_size, frame_size
                    )
                    
                    # Extract keypoints (17 keypoints * 3 values each)
                    keypoints = []
                    for i in range(17):
                        kp_start_idx = 5 + i * 3
                        if kp_start_idx + 2 < len(detection):
                            kp_x = (detection[kp_start_idx] / self.inference_engine.input_width) * frame_width
                            kp_y = (detection[kp_start_idx + 1] / self.inference_engine.input_height) * frame_height
                            kp_conf = detection[kp_start_idx + 2]
                            
                            # Ensure keypoint coordinates are within frame bounds using utils
                            kp_x, kp_y = ensure_frame_bounds((kp_x, kp_y), (frame_width, frame_height))
                            
                            keypoint = Keypoint(
                                x=float(kp_x),
                                y=float(kp_y),
                                confidence=float(kp_conf),
                                visible=kp_conf > self.keypoint_threshold
                            )
                            keypoints.append(keypoint)
                        else:
                            # Add dummy keypoint if data is missing
                            keypoints.append(Keypoint(0.0, 0.0, 0.0, False))
                    
                    # Create pose detection
                    pose_detection = PoseDetection(
                        bbox=(x1, y1, x2, y2),
                        keypoints=keypoints,
                        confidence=float(confidence)
                    )
                    detections.append(pose_detection)
        
        # Apply NMS to remove duplicate detections
        detections = self.apply_nms(detections, iou_threshold=0.5)
        
        return detections
    
    def infer_frame(self, frame: np.ndarray) -> Tuple[List[PoseDetection], np.ndarray]:
        """
        Run pose inference on a single frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (pose_detections, annotated_frame)
        """
        try:
            # Update FPS counter
            current_fps = self.fps_counter.update()
            
            # Preprocess frame and run inference using engine
            input_tensor = self.inference_engine.preprocess_frame(frame)
            outputs = self.inference_engine.infer(input_tensor)
            
            # Postprocess detections
            pose_detections = self.postprocess_detections(outputs, frame.shape[:2])
            
            # Draw annotations
            annotated_frame = self.draw_poses(frame.copy(), pose_detections)
            
            return pose_detections, annotated_frame
            
        except Exception as e:
            logger.error(f"Pose inference failed: {e}")
            return [], frame.copy()
    
    def get_human_poses(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get human pose information for AI referee system
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of dictionaries containing pose information
        """
        pose_detections, _ = self.infer_frame(frame)
        
        poses_info = []
        for i, pose in enumerate(pose_detections):
            # Convert keypoints to dictionary format
            keypoints_dict = {}
            for j, keypoint in enumerate(pose.keypoints):
                keypoint_name = self.COCO_KEYPOINT_NAMES[j]
                keypoints_dict[keypoint_name] = {
                    'x': keypoint.x,
                    'y': keypoint.y,
                    'confidence': keypoint.confidence,
                    'visible': keypoint.visible
                }
            
            pose_info = {
                'person_id': i,
                'bbox': pose.bbox,
                'center': pose.center,
                'confidence': pose.confidence,
                'avg_keypoint_confidence': pose.avg_keypoint_confidence,
                'keypoints': keypoints_dict,
                'keypoints_array': [(kp.x, kp.y, kp.confidence) for kp in pose.keypoints]
            }
            poses_info.append(pose_info)
        
        return poses_info
    
    def draw_poses(self, frame: np.ndarray, poses: List[PoseDetection]) -> np.ndarray:
        """
        Draw pose detections on frame
        
        Args:
            frame: Input frame
            poses: List of pose detections
            
        Returns:
            Annotated frame
        """
        for i, pose in enumerate(poses):
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in pose.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            conf_text = f"Person {i}: {pose.confidence:.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw keypoints
            for j, keypoint in enumerate(pose.keypoints):
                if keypoint.visible and keypoint.confidence > self.keypoint_threshold:
                    x, y = int(keypoint.x), int(keypoint.y)
                    # Draw keypoint
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                    # Draw keypoint name (optional, for debugging)
                    # cv2.putText(frame, str(j), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw skeleton
            for connection in self.COCO_SKELETON:
                kp1_idx, kp2_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-based index
                if (0 <= kp1_idx < len(pose.keypoints) and 0 <= kp2_idx < len(pose.keypoints)):
                    kp1 = pose.keypoints[kp1_idx]
                    kp2 = pose.keypoints[kp2_idx]
                    
                    if (kp1.visible and kp2.visible and 
                        kp1.confidence > self.keypoint_threshold and 
                        kp2.confidence > self.keypoint_threshold):
                        
                        pt1 = (int(kp1.x), int(kp1.y))
                        pt2 = (int(kp2.x), int(kp2.y))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        
        # Draw FPS and detection count using utils
        try:
            from .utils.openvino_utils import draw_fps_info, draw_detection_count
        except ImportError:
            from utils.openvino_utils import draw_fps_info, draw_detection_count
        
        frame = draw_fps_info(frame, self.fps_counter.get_fps())
        frame = draw_detection_count(frame, len(poses), "Persons")
        
        return frame
    
    # FPS calculation now handled by FPSCounter utility


# Optimized Pose Model class for compatibility with existing UI code
class OptimizedPoseModel(BaseOptimizedModel):
    """Wrapper class for compatibility with existing UI code"""
    
    def __init__(self, model_path: str, device: str = "CPU"):
        super().__init__(model_path, device)
        self.tracker = PoseTracker(model_path, self.device_enum)
    
    def infer_frame(self, frame: np.ndarray) -> np.ndarray:
        """Infer frame and return annotated result"""
        _, annotated_frame = self.tracker.infer_frame(frame)
        return annotated_frame
    
    def get_human_poses(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Get human pose information for AI referee"""
        return self.tracker.get_human_poses(frame)


if __name__ == "__main__":
    # Example usage for AI referee
    model_path = "models/ov_models/yolov8s-pose_openvino_model/yolov8s-pose.xml"
    tracker = PoseTracker(model_path, DeviceType.CPU)
    
    # Process video
    cap = cv2.VideoCapture("./data/ky.mov")
    
    print("Pose Tracker with OpenVINO Inference")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Get human poses for AI referee
        poses_info = tracker.get_human_poses(frame)
        
        # Display results
        pose_detections, annotated_frame = tracker.infer_frame(frame)
        
        # Print pose information for AI referee
        for pose in poses_info:
            print(f"Person {pose['person_id']}: Confidence {pose['confidence']:.2f}, "
                  f"Center at ({pose['center'][0]:.1f}, {pose['center'][1]:.1f}), "
                  f"Avg keypoint conf: {pose['avg_keypoint_confidence']:.2f}")
        
        cv2.imshow("Pose Detection with OpenVINO", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()