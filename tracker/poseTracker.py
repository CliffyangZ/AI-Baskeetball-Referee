"""
Pose Tracker with OpenVINO Runtime Inference

This module implements human pose estimation and tracking using OpenVINO for
optimized inference with support for multiple pose estimation models.
"""

import numpy as np
import cv2
import openvino as ov
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import math

# Import utilities
from utils.KalmanFilter import BasketballKalmanFilter
from utils.image_utils import bgr_to_rgb, frame_to_base64_png

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported OpenVINO device types"""
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    AUTO = "AUTO"


class PoseModel(Enum):
    """Supported pose estimation models"""
    OPENPOSE = "openpose"
    MOVENET = "movenet"
    YOLOV8_POSE = "yolov8_pose"
    HRNET = "hrnet"


@dataclass
class Keypoint:
    """Single keypoint with coordinates and confidence"""
    x: float
    y: float
    confidence: float = 0.0


@dataclass
class Pose:
    """Human pose with keypoints"""
    keypoints: List[Keypoint]
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    confidence: float = 0.0
    person_id: Optional[int] = None


@dataclass
class PoseTrack:
    """Pose track for multi-person tracking"""
    track_id: int
    pose: Pose
    state: str = "active"  # active, lost, removed
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    kalman_filter: Optional[BasketballKalmanFilter] = None
    
    def __post_init__(self):
        if self.kalman_filter is None and self.pose.bbox is not None:
            self.kalman_filter = BasketballKalmanFilter()
            # Initialize with center position and zero velocity
            cx = (self.pose.bbox[0] + self.pose.bbox[2]) / 2
            cy = (self.pose.bbox[1] + self.pose.bbox[3]) / 2
            self.kalman_filter.initialize(np.array([cx, cy, 0, 0]))


class PoseTracker:
    """
    Pose Tracker using OpenVINO Runtime
    
    Features:
    - OpenVINO optimized inference
    - Multiple pose model support
    - Multi-person tracking
    - Keypoint smoothing with Kalman filtering
    """
    
    # COCO pose keypoint indices
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

# Export COCO keypoint names for compatibility
COCO_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class PoseTracker:
    """
    Pose Tracker using OpenVINO Runtime
    
    Features:
    - OpenVINO optimized inference
    - Multiple pose model support
    - Multi-person tracking
    - Keypoint smoothing with Kalman filtering
    """
    
    # COCO pose keypoint indices
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # Skeleton connections for drawing
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (11, 12), (5, 11), (6, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    def __init__(
        self,
        model_path: str,
        device: DeviceType = DeviceType.CPU,
        model_type: PoseModel = PoseModel.YOLOV8_POSE,
        confidence_threshold: float = 0.3,
        max_time_lost: int = 30
    ):
        """
        Initialize Pose Tracker
        
        Args:
            model_path: Path to OpenVINO IR model (.xml file)
            device: OpenVINO device type
            model_type: Type of pose estimation model
            confidence_threshold: Minimum confidence for pose detection
            max_time_lost: Maximum frames to keep lost tracks
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.max_time_lost = max_time_lost
        
        # Initialize OpenVINO
        self._init_openvino()
        
        # Tracking state
        self.tracks: List[PoseTrack] = []
        self.track_id_counter = 0
        self.frame_count = 0
        
        logger.info(f"PoseTracker initialized with device: {device.value}, model: {model_type.value}")
    
    def _init_openvino(self):
        """Initialize OpenVINO Core and compile model"""
        try:
            # Initialize OpenVINO Core
            self.core = ov.Core()
            
            # Read model
            logger.info(f"Loading model from: {self.model_path}")
            self.model = self.core.read_model(self.model_path)
            
            # Compile model
            self.compiled_model = self.core.compile_model(
                model=self.model, 
                device_name=self.device.value
            )
            
            # Get input/output info
            self.input_layer = self.compiled_model.input(0)
            self.output_layers = [self.compiled_model.output(i) for i in range(len(self.compiled_model.outputs))]
            
            # Get input shape
            self.input_shape = self.input_layer.shape
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
            
            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
            logger.info(f"Number of outputs: {len(self.output_layers)}")
            
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
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_frame, (self.input_width, self.input_height))
        
        # Normalize based on model type
        if self.model_type in [PoseModel.YOLOV8_POSE, PoseModel.MOVENET]:
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
        else:
            # Normalize to [-1, 1] for some models
            normalized = (resized.astype(np.float32) / 127.5) - 1.0
        
        # Add batch dimension and transpose to NCHW
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess_yolov8_pose(
        self, 
        outputs: List[np.ndarray], 
        frame_shape: Tuple[int, int]
    ) -> List[Pose]:
        """Postprocess YOLOv8 pose model outputs"""
        poses = []
        frame_height, frame_width = frame_shape
        
        # Scale factors
        scale_x = frame_width / self.input_width
        scale_y = frame_height / self.input_height
        
        # YOLOv8 pose output format: [batch, 56, 8400]
        # 56 = 4 (bbox) + 1 (conf) + 51 (17 keypoints * 3)
        if len(outputs) > 0:
            output = outputs[0]
            if len(output.shape) == 3:
                output = output[0]  # Remove batch dimension
            
            # Transpose to [8400, 56]
            if output.shape[0] == 56:
                output = output.T
            
            for detection in output:
                if len(detection) >= 56:
                    # Extract bbox and confidence
                    cx, cy, w, h, conf = detection[:5]
                    
                    if conf < self.confidence_threshold:
                        continue
                    
                    # Convert center format to corner format
                    x1 = (cx - w/2) * scale_x
                    y1 = (cy - h/2) * scale_y
                    x2 = (cx + w/2) * scale_x
                    y2 = (cy + h/2) * scale_y
                    
                    # Clamp to frame boundaries
                    x1 = max(0, min(x1, frame_width))
                    y1 = max(0, min(y1, frame_height))
                    x2 = max(0, min(x2, frame_width))
                    y2 = max(0, min(y2, frame_height))
                    
                    # Extract keypoints
                    keypoints = []
                    for i in range(17):  # 17 COCO keypoints
                        kp_x = detection[5 + i*3] * scale_x
                        kp_y = detection[5 + i*3 + 1] * scale_y
                        kp_conf = detection[5 + i*3 + 2]
                        
                        # Clamp keypoint coordinates
                        kp_x = max(0, min(kp_x, frame_width))
                        kp_y = max(0, min(kp_y, frame_height))
                        
                        keypoints.append(Keypoint(kp_x, kp_y, kp_conf))
                    
                    poses.append(Pose(
                        keypoints=keypoints,
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf)
                    ))
        
        return poses
    
    def postprocess_openpose(
        self, 
        outputs: List[np.ndarray], 
        frame_shape: Tuple[int, int]
    ) -> List[Pose]:
        """Postprocess OpenPose model outputs"""
        poses = []
        frame_height, frame_width = frame_shape
        
        if len(outputs) > 0:
            # OpenPose output: [1, 57, H, W] where 57 = 18 keypoints + 19 PAFs
            heatmaps = outputs[0]
            if len(heatmaps.shape) == 4:
                heatmaps = heatmaps[0]  # Remove batch dimension
            
            # Extract keypoint heatmaps (first 18 channels)
            keypoint_heatmaps = heatmaps[:18]
            
            # Find peaks in heatmaps
            keypoints = []
            for i, heatmap in enumerate(keypoint_heatmaps):
                # Find maximum in heatmap
                max_val = np.max(heatmap)
                if max_val > self.confidence_threshold:
                    max_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    y, x = max_loc
                    
                    # Scale to original frame size
                    scaled_x = x * frame_width / heatmap.shape[1]
                    scaled_y = y * frame_height / heatmap.shape[0]
                    
                    keypoints.append(Keypoint(scaled_x, scaled_y, max_val))
                else:
                    keypoints.append(Keypoint(0, 0, 0))
            
            if len(keypoints) > 0:
                # Calculate bounding box from visible keypoints
                visible_kps = [kp for kp in keypoints if kp.confidence > 0.1]
                if visible_kps:
                    xs = [kp.x for kp in visible_kps]
                    ys = [kp.y for kp in visible_kps]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    
                    # Add padding
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame_width, x2 + padding)
                    y2 = min(frame_height, y2 + padding)
                    
                    avg_conf = np.mean([kp.confidence for kp in visible_kps])
                    
                    poses.append(Pose(
                        keypoints=keypoints,
                        bbox=(x1, y1, x2, y2),
                        confidence=avg_conf
                    ))
        
        return poses
    
    def postprocess_detections(
        self, 
        outputs: List[np.ndarray], 
        frame_shape: Tuple[int, int]
    ) -> List[Pose]:
        """
        Postprocess model outputs based on model type
        
        Args:
            outputs: Raw model outputs
            frame_shape: Original frame shape (height, width)
            
        Returns:
            List of pose detections
        """
        if self.model_type == PoseModel.YOLOV8_POSE:
            return self.postprocess_yolov8_pose(outputs, frame_shape)
        elif self.model_type == PoseModel.OPENPOSE:
            return self.postprocess_openpose(outputs, frame_shape)
        else:
            # Generic postprocessing for other models
            return self.postprocess_yolov8_pose(outputs, frame_shape)
    
    def calculate_pose_similarity(self, pose1: Pose, pose2: Pose) -> float:
        """Calculate similarity between two poses using keypoint distances"""
        if not pose1.keypoints or not pose2.keypoints:
            return 0.0
        
        if len(pose1.keypoints) != len(pose2.keypoints):
            return 0.0
        
        total_distance = 0.0
        valid_keypoints = 0
        
        for kp1, kp2 in zip(pose1.keypoints, pose2.keypoints):
            if kp1.confidence > 0.1 and kp2.confidence > 0.1:
                distance = math.sqrt((kp1.x - kp2.x)**2 + (kp1.y - kp2.y)**2)
                total_distance += distance
                valid_keypoints += 1
        
        if valid_keypoints == 0:
            return 0.0
        
        # Normalize by average distance and convert to similarity
        avg_distance = total_distance / valid_keypoints
        similarity = 1.0 / (1.0 + avg_distance / 100.0)  # Normalize by 100 pixels
        
        return similarity
    
    def calculate_bbox_iou(self, bbox1: Tuple[float, float, float, float], 
                          bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_tracks(self, poses: List[Pose]) -> List[PoseTrack]:
        """
        Update pose tracks
        
        Args:
            poses: List of detected poses
            
        Returns:
            List of active pose tracks
        """
        self.frame_count += 1
        
        # Predict track positions using Kalman filter
        for track in self.tracks:
            if track.kalman_filter and track.kalman_filter.initialized:
                try:
                    predicted_pos = track.kalman_filter.predict()
                    if predicted_pos is not None and len(predicted_pos) >= 2 and track.pose.bbox:
                        # Update bbox center based on prediction
                        cx, cy = predicted_pos[0], predicted_pos[1]
                        w = track.pose.bbox[2] - track.pose.bbox[0]
                        h = track.pose.bbox[3] - track.pose.bbox[1]
                        track.pose.bbox = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
                except:
                    pass  # Continue with previous bbox if prediction fails
        
        # Associate poses with tracks
        active_tracks = [t for t in self.tracks if t.state == "active"]
        
        # Simple association based on bbox IoU and pose similarity
        matches = []
        unmatched_tracks = list(range(len(active_tracks)))
        unmatched_poses = list(range(len(poses)))
        
        # Calculate similarity matrix
        if active_tracks and poses:
            similarity_matrix = np.zeros((len(active_tracks), len(poses)))
            
            for t, track in enumerate(active_tracks):
                for p, pose in enumerate(poses):
                    # Combine bbox IoU and pose similarity
                    bbox_sim = 0.0
                    if track.pose.bbox and pose.bbox:
                        bbox_sim = self.calculate_bbox_iou(track.pose.bbox, pose.bbox)
                    
                    pose_sim = self.calculate_pose_similarity(track.pose, pose)
                    
                    # Weighted combination (more weight on bbox for better tracking)
                    similarity_matrix[t, p] = 0.6 * bbox_sim + 0.4 * pose_sim
            
            # Improved matching with sorted similarity scores
            potential_matches = []
            for t in range(len(active_tracks)):
                for p in range(len(poses)):
                    if similarity_matrix[t, p] > 0.2:  # Lower threshold for better association
                        potential_matches.append((similarity_matrix[t, p], t, p))
            
            # Sort by similarity score descending
            potential_matches.sort(reverse=True)
            
            # Assign matches greedily, starting with highest similarity
            for sim_score, t, p in potential_matches:
                if t in unmatched_tracks and p in unmatched_poses:
                    matches.append((t, p))
                    unmatched_tracks.remove(t)
                    unmatched_poses.remove(p)
        
        # Update matched tracks
        for track_idx, pose_idx in matches:
            track = active_tracks[track_idx]
            pose = poses[pose_idx]
            
            track.pose = pose
            track.hits += 1
            track.time_since_update = 0
            
            # Update Kalman filter
            if track.kalman_filter and track.kalman_filter.initialized and pose.bbox:
                try:
                    cx = (pose.bbox[0] + pose.bbox[2]) / 2
                    cy = (pose.bbox[1] + pose.bbox[3]) / 2
                    track.kalman_filter.update(np.array([cx, cy]))
                except:
                    pass
        
        # Create new tracks for unmatched poses
        for pose_idx in unmatched_poses:
            pose = poses[pose_idx]
            new_track = PoseTrack(
                track_id=self.track_id_counter,
                pose=pose,
                hits=1
            )
            self.tracks.append(new_track)
            self.track_id_counter += 1
        
        # Update track states and remove old tracks
        for track in self.tracks[:]:
            track.age += 1
            if track.time_since_update == 0:
                continue
            
            track.time_since_update += 1
            
            if track.state == "active" and track.time_since_update > 1:
                track.state = "lost"
            elif track.time_since_update > self.max_time_lost:
                self.tracks.remove(track)
        
        # Return active tracks
        return [t for t in self.tracks if t.state == "active"]
    
    def infer_frame(self, frame: np.ndarray) -> Tuple[List[PoseTrack], np.ndarray]:
        """
        Run pose inference on a single frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (active_tracks, annotated_frame)
        """
        # Preprocess frame
        input_tensor = self.preprocess_frame(frame)
        
        # Run inference
        try:
            results = self.compiled_model([input_tensor])
            outputs = [results[layer] for layer in self.output_layers]
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return [], frame
        
        # Postprocess detections
        poses = self.postprocess_detections(outputs, frame.shape[:2])
        
        # Update tracks
        active_tracks = self.update_tracks(poses)
        
        # Draw clean annotations without trails
        annotated_frame = self.draw_clean_poses(frame.copy(), active_tracks)
        
        return active_tracks, annotated_frame
    
    def draw_clean_poses(self, frame: np.ndarray, tracks: List[PoseTrack]) -> np.ndarray:
        """Draw only current pose keypoints and skeleton without trails or history"""
        # Ensure we're working with a fresh copy of the frame
        clean_frame = frame.copy()
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for i, track in enumerate(tracks):
            color = colors[i % len(colors)]
            pose = track.pose
            
            # Draw clean bounding box
            if pose.bbox:
                x1, y1, x2, y2 = map(int, pose.bbox)
                
                # Ensure coordinates are valid
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(clean_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw track ID with background for better visibility
                    label = f"Person {track.track_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw background rectangle for text
                    cv2.rectangle(clean_frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    
                    # Draw text
                    cv2.putText(clean_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw clean keypoints
            for j, keypoint in enumerate(pose.keypoints):
                if keypoint.confidence > 0.5:  # Higher threshold for cleaner display
                    x, y = int(keypoint.x), int(keypoint.y)
                    # Draw keypoint with outline for better visibility
                    cv2.circle(clean_frame, (x, y), 5, (0, 0, 0), -1)  # Black outline
                    cv2.circle(clean_frame, (x, y), 4, color, -1)  # Colored center
            
            # Draw clean skeleton connections
            for connection in self.SKELETON_CONNECTIONS:
                kp1_idx, kp2_idx = connection
                if (kp1_idx < len(pose.keypoints) and kp2_idx < len(pose.keypoints)):
                    kp1 = pose.keypoints[kp1_idx]
                    kp2 = pose.keypoints[kp2_idx]
                    
                    if kp1.confidence > 0.5 and kp2.confidence > 0.5:
                        pt1 = (int(kp1.x), int(kp1.y))
                        pt2 = (int(kp2.x), int(kp2.y))
                        # Draw line with outline for better visibility
                        cv2.line(clean_frame, pt1, pt2, (0, 0, 0), 4)  # Black outline
                        cv2.line(clean_frame, pt1, pt2, color, 2)  # Colored line
        
        return clean_frame
    
    def draw_poses(self, frame: np.ndarray, tracks: List[PoseTrack]) -> np.ndarray:
        """Draw pose keypoints and skeleton on frame (kept for compatibility)"""
        return self.draw_clean_poses(frame, tracks)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get OpenVINO device information"""
        try:
            available_devices = self.core.available_devices
            device_info = {
                "available_devices": available_devices,
                "current_device": self.device.value,
                "model_path": str(self.model_path),
                "model_type": self.model_type.value,
                "input_shape": self.input_shape
            }
            return device_info
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}


# Optimized Pose Model class for compatibility with existing UI code
class OptimizedPoseModel:
    """Wrapper class for compatibility with existing UI code"""
    
    def __init__(self, model_path: str, device: str = "CPU", model_type: str = "yolov8_pose"):
        device_enum = DeviceType(device.upper())
        model_type_enum = PoseModel(model_type.lower())
        self.tracker = PoseTracker(model_path, device_enum, model_type_enum)
    
    def infer_frame(self, frame: np.ndarray) -> np.ndarray:
        """Infer frame and return annotated result"""
        _, annotated_frame = self.tracker.infer_frame(frame)
        return annotated_frame
    
    def get_poses(self, frame: np.ndarray) -> List[PoseTrack]:
        """Get active pose tracks for a frame"""
        tracks, _ = self.tracker.infer_frame(frame)
        return tracks


if __name__ == "__main__":
    # Example usage
    model_path = "models/ov_models/yolov8s-pose_openvino_model/yolov8s-pose.xml"
    tracker = PoseTracker(model_path, DeviceType.CPU, PoseModel.YOLOV8_POSE)
    
    # Process video
    cap = cv2.VideoCapture("./data/video/travel.mov")  # or video file path
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        tracks, annotated_frame = tracker.infer_frame(frame)
        
        cv2.imshow("Pose Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()