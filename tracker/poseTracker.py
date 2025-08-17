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
import sys
import os

# Add project root to path to fix imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
# Import directly from project root utils
from tracker.utils.image_utils import bgr_to_rgb, frame_to_base64_png
from tracker.utils.algorithm import calculate_iou, calculate_iou_matrix, hungarian_matching, apply_nms
from tracker.utils.openvino_utils import (
    DeviceType, OpenVINOInferenceEngine, FPSCounter, 
    normalize_coordinates, ensure_frame_bounds, BaseOptimizedModel
)
from tracker.utils.KalmanFilter import BasketballKalmanFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# DeviceType now imported from openvino_utils


class PoseModel(Enum):
    """Supported pose model types"""
    YOLOV8_POSE = "yolov8_pose"
    YOLOV11_POSE = "yolov11_pose"


class TrackState(Enum):
    """Track state enumeration for pose tracking"""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


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


class PoseTrack:
    """
    Single pose track class for ByteTrack algorithm
    Manages individual person tracking with pose-specific features
    """
    
    track_id_count = 0
    
    def __init__(self, detection: PoseDetection, track_id: Optional[int] = None):
        """Initialize pose track with detection"""
        self.detection = detection
        self.bbox = detection.bbox
        self.center = detection.center
        self.keypoints = detection.keypoints
        self.confidence = detection.confidence
        
        # Track management
        self.track_id = track_id if track_id is not None else self._next_id()
        self.state = TrackState.NEW
        self.is_activated = False
        
        # Tracking history
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0
        
        # Motion prediction for center point
        self.kalman_filter = BasketballKalmanFilter()
        
        # Pose-specific tracking
        self.pose_history = []  # Store recent pose detections
        self.keypoint_history = []  # Store keypoint trajectories
        self.pose_similarity_threshold = 0.3
        
    @classmethod
    def _next_id(cls):
        """Get next track ID"""
        cls.track_id_count += 1
        return cls.track_id_count
    
    @classmethod
    def reset_id(cls):
        """Reset track ID counter"""
        cls.track_id_count = 0
    
    def activate(self, frame_id: int):
        """Activate pose track"""
        self.track_id = self._next_id()
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
        # Initialize Kalman filter with center point
        if self.kalman_filter:
            cx, cy = self.center
            self.kalman_filter.initialize(np.array([cx, cy, 0, 0]))
    
    def re_activate(self, detection: PoseDetection, frame_id: int):
        """Re-activate lost pose track"""
        self.update(detection, frame_id)
        self.state = TrackState.TRACKED
        self.is_activated = True
    
    def update(self, detection: PoseDetection, frame_id: int):
        """Update pose track with new detection"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        # Update detection info
        self.detection = detection
        self.bbox = detection.bbox
        self.center = detection.center
        self.keypoints = detection.keypoints
        self.confidence = detection.confidence
        
        # Update pose history
        self.pose_history.append(detection)
        if len(self.pose_history) > 10:  # Keep last 10 poses
            self.pose_history.pop(0)
        
        # Update keypoint history
        self.keypoint_history.append([kp for kp in detection.keypoints])
        if len(self.keypoint_history) > 10:
            self.keypoint_history.pop(0)
        
        # Update Kalman filter with center point
        if self.kalman_filter and self.kalman_filter.initialized:
            try:
                cx, cy = detection.center
                self.kalman_filter.update(np.array([cx, cy]))
            except:
                pass
        
        self.state = TrackState.TRACKED
    
    def predict(self):
        """Predict next position using Kalman filter"""
        if self.kalman_filter and self.kalman_filter.initialized:
            try:
                predicted_pos = self.kalman_filter.predict()
                if predicted_pos is not None and len(predicted_pos) >= 4:
                    cx, cy, vx, vy = predicted_pos[:4]
                    # Update predicted center
                    self.center = (cx, cy)
                    # Update predicted bbox (maintain size)
                    w = self.bbox[2] - self.bbox[0]
                    h = self.bbox[3] - self.bbox[1]
                    self.bbox = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
            except:
                pass
    
    def mark_lost(self):
        """Mark pose track as lost"""
        self.state = TrackState.LOST
    
    def mark_removed(self):
        """Mark pose track as removed"""
        self.state = TrackState.REMOVED
    
    def calculate_pose_similarity(self, other_detection: PoseDetection) -> float:
        """Calculate pose similarity based on keypoint positions"""
        if not self.keypoints or not other_detection.keypoints:
            return 0.0
        
        # Calculate weighted keypoint distance
        total_distance = 0.0
        valid_pairs = 0
        
        # Keypoint importance weights (higher for more stable keypoints)
        keypoint_weights = [
            0.8,   # nose
            0.6, 0.6,  # eyes
            0.5, 0.5,  # ears
            1.0, 1.0,  # shoulders
            0.9, 0.9,  # elbows
            0.8, 0.8,  # wrists
            1.0, 1.0,  # hips
            0.9, 0.9,  # knees
            0.7, 0.7   # ankles
        ]
        
        for i, (kp1, kp2) in enumerate(zip(self.keypoints, other_detection.keypoints)):
            if (kp1.visible and kp2.visible and 
                kp1.confidence > 0.3 and kp2.confidence > 0.3):
                
                distance = np.sqrt((kp1.x - kp2.x)**2 + (kp1.y - kp2.y)**2)
                weight = keypoint_weights[i] if i < len(keypoint_weights) else 0.5
                total_distance += distance * weight
                valid_pairs += weight
        
        if valid_pairs == 0:
            return 0.0
        
        # Normalize by number of valid pairs and convert to similarity
        avg_distance = total_distance / valid_pairs
        # Convert distance to similarity (closer = higher similarity)
        similarity = max(0.0, 1.0 - (avg_distance / 200.0))  # 200 pixels as max distance
        
        return similarity


def calculate_pose_similarity_matrix(tracks: List[PoseTrack], detections: List[PoseDetection]) -> np.ndarray:
    """
    Calculate pose similarity matrix between tracks and detections
    Combines IoU and keypoint similarity for robust matching
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    
    similarity_matrix = np.zeros((len(tracks), len(detections)))
    
    for t, track in enumerate(tracks):
        for d, detection in enumerate(detections):
            # Calculate IoU similarity
            iou = calculate_iou(track.bbox, detection.bbox)
            
            # Calculate pose similarity
            pose_sim = track.calculate_pose_similarity(detection)
            
            # Combined similarity (weighted average)
            combined_similarity = 0.4 * iou + 0.6 * pose_sim
            similarity_matrix[t, d] = combined_similarity
    
    return similarity_matrix


class PoseBYTETracker:
    """
    ByteTrack tracker for pose detection with pose-specific optimizations
    Implements ByteTrack algorithm with keypoint similarity matching
    """
    
    def __init__(self, 
                 high_thresh: float = 0.6,
                 low_thresh: float = 0.1,
                 match_thresh: float = 0.8,
                 track_buffer: int = 30,
                 frame_rate: int = 30):
        """
        Initialize Pose ByteTracker
        
        Args:
            high_thresh: High confidence threshold for first association
            low_thresh: Low confidence threshold for second association
            match_thresh: Similarity threshold for track association
            track_buffer: Number of frames to keep lost tracks
            frame_rate: Video frame rate
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        
        # Track management
        self.tracked_stracks = []  # Active tracks
        self.lost_stracks = []     # Lost tracks
        self.removed_stracks = []  # Removed tracks
        
        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        
        logger.info(f"PoseBYTETracker initialized: high_thresh={high_thresh}, low_thresh={low_thresh}")
    
    def update(self, detections: List[PoseDetection]) -> List[PoseTrack]:
        """
        Update tracker with new pose detections
        
        Args:
            detections: List of pose detections
            
        Returns:
            List of active pose tracks
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Separate detections by confidence
        high_dets = [det for det in detections if det.confidence >= self.high_thresh]
        low_dets = [det for det in detections if self.low_thresh <= det.confidence < self.high_thresh]
        
        # Predict current positions of existing tracks
        for track in self.tracked_stracks + self.lost_stracks:
            track.predict()
        
        # First association: high confidence detections with tracked tracks
        if len(high_dets) > 0 and len(self.tracked_stracks) > 0:
            # Calculate combined similarity matrix (IoU + pose similarity)
            similarity_matrix = calculate_pose_similarity_matrix(self.tracked_stracks, high_dets)
            
            # Hungarian matching
            matches, unmatched_tracks, unmatched_dets = hungarian_matching(
                similarity_matrix, self.match_thresh
            )
            
            # Update matched tracks
            for track_idx, det_idx in matches:
                track = self.tracked_stracks[track_idx]
                detection = high_dets[det_idx]
                track.update(detection, self.frame_id)
                activated_stracks.append(track)
            
            # Handle unmatched tracks from first association
            for track_idx in unmatched_tracks:
                track = self.tracked_stracks[track_idx]
                if track.state != TrackState.LOST:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            # No high confidence detections or no tracked tracks
            unmatched_dets = list(range(len(high_dets)))
            for track in self.tracked_stracks:
                track.mark_lost()
                lost_stracks.append(track)
        
        # Second association: low confidence detections with lost tracks
        if len(low_dets) > 0 and len(lost_stracks) > 0:
            similarity_matrix = calculate_pose_similarity_matrix(lost_stracks, low_dets)
            matches, unmatched_lost, unmatched_low_dets = hungarian_matching(
                similarity_matrix, 0.4  # Lower threshold for lost tracks
            )
            
            # Re-activate matched lost tracks
            for track_idx, det_idx in matches:
                track = lost_stracks[track_idx]
                detection = low_dets[det_idx]
                track.re_activate(detection, self.frame_id)
                refind_stracks.append(track)
            
            # Update remaining lost tracks
            remaining_lost = []
            for track_idx in unmatched_lost:
                track = lost_stracks[track_idx]
                if self.frame_id - track.frame_id <= self.max_time_lost:
                    remaining_lost.append(track)
                else:
                    track.mark_removed()
                    removed_stracks.append(track)
            lost_stracks = remaining_lost
        
        # Initialize new tracks for remaining unmatched detections
        for det_idx in unmatched_dets:
            detection = high_dets[det_idx]
            track = PoseTrack(detection)
            track.activate(self.frame_id)
            activated_stracks.append(track)
        
        # Update track lists
        self.tracked_stracks = activated_stracks + refind_stracks
        
        # Remove old lost tracks
        self.lost_stracks = []
        for track in lost_stracks:
            if self.frame_id - track.frame_id <= self.max_time_lost:
                self.lost_stracks.append(track)
            else:
                track.mark_removed()
                removed_stracks.append(track)
        
        # Update removed tracks
        self.removed_stracks.extend(removed_stracks)
        
        return self.tracked_stracks
    
    def reset(self):
        """Reset tracker state"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        PoseTrack.reset_id()


class PoseTracker:
    """
    Enhanced Pose Tracker using OpenVINO Runtime with ByteTrack Algorithm
    
    Features:
    - OpenVINO optimized inference
    - YOLO pose model support (YOLOv8-Pose, YOLOv11-Pose)
    - COCO 17-keypoint format
    - ByteTrack multi-person tracking with pose similarity
    - Real-time pose detection and tracking
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
        keypoint_threshold: float = 0.3,
        enable_tracking: bool = True
    ):
        """
        Initialize Enhanced Pose Tracker for AI referee
        
        Args:
            model_path: Path to OpenVINO IR model (.xml file)
            device: OpenVINO device type
            pose_model: Type of pose model
            confidence_threshold: Minimum confidence for pose detection
            keypoint_threshold: Minimum confidence for keypoint visibility
            enable_tracking: Enable ByteTrack multi-person tracking
        """
        self.model_path = Path(model_path)
        self.device = device
        self.pose_model = pose_model
        self.confidence_threshold = confidence_threshold
        self.keypoint_threshold = keypoint_threshold
        self.enable_tracking = enable_tracking
        
        # Initialize OpenVINO engine
        # Convert device to string if it's an enum
        device_str = device.value if hasattr(device, 'value') else str(device)
        self.inference_engine = OpenVINOInferenceEngine(model_path, device_str)
        
        # Performance monitoring
        self.fps_counter = FPSCounter()
        
        # Initialize ByteTrack tracker if enabled
        if self.enable_tracking:
            self.pose_tracker = PoseBYTETracker(
                high_thresh=confidence_threshold,
                low_thresh=max(0.1, confidence_threshold - 0.3),
                match_thresh=0.7,
                track_buffer=30
            )
        else:
            self.pose_tracker = None
        
        # Safe logging that works with both enum and string
        device_value = device.value if hasattr(device, 'value') else device
        pose_model_value = pose_model.value if hasattr(pose_model, 'value') else pose_model
        tracking_status = "enabled" if enable_tracking else "disabled"
        logger.info(f"PoseTracker initialized with device: {device_value}, model: {pose_model_value}, tracking: {tracking_status}")
    
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
        Run pose inference and tracking on a single frame
        
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
            
            # Apply tracking if enabled
            if self.enable_tracking and self.pose_tracker:
                tracked_poses = self.pose_tracker.update(pose_detections)
                # Update pose detections with track IDs
                pose_detections = [track.detection for track in tracked_poses]
                for i, track in enumerate(tracked_poses):
                    if i < len(pose_detections):
                        pose_detections[i].person_id = track.track_id
            
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
            List of dictionaries containing pose information with consistent tracking IDs
        """
        if self.enable_tracking and self.pose_tracker:
            # Run inference to update tracker
            pose_detections, _ = self.infer_frame(frame)
            
            # Get tracked poses with consistent IDs
            poses_info = []
            for track in self.pose_tracker.tracked_stracks:
                # Convert keypoints to dictionary format
                keypoints_dict = {}
                for j, keypoint in enumerate(track.keypoints):
                    if j < len(self.COCO_KEYPOINT_NAMES):
                        keypoint_name = self.COCO_KEYPOINT_NAMES[j]
                        keypoints_dict[keypoint_name] = {
                            'x': keypoint.x,
                            'y': keypoint.y,
                            'confidence': keypoint.confidence,
                            'visible': keypoint.visible
                        }
                
                pose_info = {
                    'person_id': track.track_id,
                    'bbox': track.bbox,
                    'center': track.center,
                    'confidence': track.confidence,
                    'avg_keypoint_confidence': track.detection.avg_keypoint_confidence,
                    'keypoints': keypoints_dict,
                    'keypoints_array': [(kp.x, kp.y, kp.confidence) for kp in track.keypoints],
                    'track_length': track.tracklet_len,
                    'frame_id': self.pose_tracker.frame_id
                }
                poses_info.append(pose_info)
            
            return poses_info
        else:
            # Use detection-only mode
            pose_detections, _ = self.infer_frame(frame)
            
            poses_info = []
            for i, pose in enumerate(pose_detections):
                # Convert keypoints to dictionary format
                keypoints_dict = {}
                for j, keypoint in enumerate(pose.keypoints):
                    if j < len(self.COCO_KEYPOINT_NAMES):
                        keypoint_name = self.COCO_KEYPOINT_NAMES[j]
                        keypoints_dict[keypoint_name] = {
                            'x': keypoint.x,
                            'y': keypoint.y,
                            'confidence': keypoint.confidence,
                            'visible': keypoint.visible
                        }
                
                pose_info = {
                    'person_id': pose.person_id,
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
            # Draw bounding box with more appealing style
            x1, y1, x2, y2 = [int(coord) for coord in pose.bbox]
            
            # Draw a more visually appealing bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            # Draw ID badge in top-left corner of bbox
            badge_width = 70
            badge_height = 25
            cv2.rectangle(frame, (x1, y1-badge_height), (x1+badge_width, y1), (0, 165, 255), -1)
            cv2.putText(frame, f"Person {pose.person_id}", (x1+5, y1-7), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw confidence as a small meter
            meter_width = 40
            meter_height = 4
            meter_x = x1 + badge_width + 5
            meter_y = y1 - 15
            # Background bar
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (100, 100, 100), -1)
            # Filled portion based on confidence
            filled_width = int(meter_width * pose.confidence)
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + filled_width, meter_y + meter_height), 
                         (0, 255, 0) if pose.confidence > 0.7 else (0, 165, 255), -1)
            
            # Draw keypoints with improved visibility
            for j, keypoint in enumerate(pose.keypoints):
                if keypoint.visible and keypoint.confidence > self.keypoint_threshold:
                    x, y = int(keypoint.x), int(keypoint.y)
                    # Draw keypoint with size based on confidence
                    radius = max(3, int(5 * keypoint.confidence))
                    # Use different colors for different keypoint types
                    if j < 5:  # Face keypoints
                        color = (255, 255, 0)  # Yellow
                    elif j < 11:  # Upper body
                        color = (0, 255, 255)  # Cyan
                    else:  # Lower body
                        color = (255, 0, 255)  # Magenta
                    
                    cv2.circle(frame, (x, y), radius, color, -1)
                    # Add white outline for better visibility
                    cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)
            
            # Draw skeleton with improved visibility
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
                        
                        # Calculate average confidence for this connection
                        avg_conf = (kp1.confidence + kp2.confidence) / 2.0
                        
                        # Determine line thickness based on confidence
                        thickness = max(1, int(3 * avg_conf))
                        
                        # Draw line with gradient color based on body part
                        if kp1_idx < 5 or kp2_idx < 5:  # Face connections
                            color = (70, 130, 180)  # Steel Blue
                        elif kp1_idx < 11 and kp2_idx < 11:  # Upper body
                            color = (32, 165, 218)  # Dodger Blue
                        else:  # Lower body
                            color = (0, 128, 255)  # Royal Blue
                        
                        # Draw the line with anti-aliasing
                        cv2.line(frame, pt1, pt2, color, thickness)
        
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