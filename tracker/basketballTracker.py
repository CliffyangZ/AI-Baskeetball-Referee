"""
Basketball Tracker with OpenVINO Runtime Inference and BYTETrack Algorithm

This module implements basketball detection and tracking using OpenVINO for
optimized inference and BYTETrack for multi-object tracking.
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


@dataclass
class Detection:
    """Basketball detection result"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0  # Basketball class


@dataclass
class Track:
    """Basketball track with BYTETrack implementation"""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    state: str = "active"  # active, lost, removed
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    kalman_filter: Optional[BasketballKalmanFilter] = None
    
    def __post_init__(self):
        if self.kalman_filter is None:
            self.kalman_filter = BasketballKalmanFilter()
            # Initialize with center position and zero velocity
            cx, cy = (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
            self.kalman_filter.initialize(np.array([cx, cy, 0, 0]))


class BasketballTracker:
    """
    Basketball Tracker using OpenVINO Runtime and BYTETrack algorithm
    
    Features:
    - OpenVINO optimized inference
    - BYTETrack multi-object tracking
    - Kalman filtering for smooth trajectories
    - Configurable confidence thresholds
    """
    
    def __init__(
        self,
        model_path: str,
        device: DeviceType,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.3,  # Lower IoU threshold for better association
        max_time_lost: int = 30
    ):
        """
        Initialize Basketball Tracker
        
        Args:
            model_path: Path to OpenVINO IR model (.xml file)
            device: OpenVINO device type
            high_thresh: High confidence threshold for first association
            low_thresh: Low confidence threshold for second association  
            match_thresh: IoU threshold for track association
            max_time_lost: Maximum frames to keep lost tracks
        """
        self.model_path = Path(model_path)
        self.device = device
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        
        # Initialize OpenVINO
        self._init_openvino()
        
        # BYTETrack state
        self.tracks: List[Track] = []
        self.track_id_counter = 0
        self.frame_count = 0
        
        logger.info(f"BasketballTracker initialized with device: {device.value}")
    
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
            self.output_layer = self.compiled_model.output(0)
            
            # Get input shape
            self.input_shape = self.input_layer.shape
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
            
            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
            
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
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess_detections(
        self, 
        outputs: np.ndarray, 
        frame_shape: Tuple[int, int]
    ) -> List[Detection]:
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
        
        # Scale factors for bbox coordinates
        scale_x = frame_width / self.input_width
        scale_y = frame_height / self.input_height
        
        # Process outputs (YOLOv8 format handling)
        logger.info(f"Raw outputs shape: {outputs.shape}")
        
        # Handle YOLOv8 output format: [1, 5, 3549] -> [1, 5, num_anchors]
        # Format: [x_center, y_center, width, height, confidence]
        if len(outputs.shape) == 3:
            batch_size, num_classes_plus_coords, num_anchors = outputs.shape
            outputs = outputs[0]  # Remove batch dimension: [5, 3549]
            
            # Transpose to [num_anchors, 5]
            outputs = outputs.transpose()  # [3549, 5]
        
        logger.info(f"Processing outputs shape: {outputs.shape}")
        
        # Process detections in YOLOv8 format
        for i, detection in enumerate(outputs):
            if len(detection) >= 5:
                x_center, y_center, width, height, conf = detection[:5]
                
                # Debug: log first few detections
                if i < 5:
                    logger.info(f"Raw detection {i}: x_center={x_center:.3f}, y_center={y_center:.3f}, w={width:.3f}, h={height:.3f}, conf={conf:.3f}")
                
                # Apply sigmoid to confidence if needed (values > 1 suggest raw logits)
                if conf > 1.0:
                    conf = 1.0 / (1.0 + np.exp(-conf))  # Sigmoid activation
                
                # Filter by confidence
                if conf >= 0.25:  # Lower threshold for better detection
                    # Convert center format to corner format
                    x1 = (x_center - width / 2) * scale_x
                    y1 = (y_center - height / 2) * scale_y
                    x2 = (x_center + width / 2) * scale_x
                    y2 = (y_center + height / 2) * scale_y
                    
                    # Clamp to frame boundaries
                    x1 = max(0, min(x1, frame_width))
                    y1 = max(0, min(y1, frame_height))
                    x2 = max(0, min(x2, frame_width))
                    y2 = max(0, min(y2, frame_height))
                    
                    # Ensure valid bounding box
                    if x2 > x1 and y2 > y1:
                        detections.append(Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=float(conf),
                            class_id=0  # Assume basketball class
                        ))
        
        return detections
    
    def calculate_iou(self, bbox1: Tuple[float, float, float, float], 
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
    
    def associate_detections_to_tracks(
        self, 
        detections: List[Detection], 
        tracks: List[Track],
        iou_threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using Hungarian algorithm
        
        Returns:
            matches: List of (track_idx, detection_idx) pairs
            unmatched_tracks: List of unmatched track indices
            unmatched_detections: List of unmatched detection indices
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(range(len(tracks))), []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t, track in enumerate(tracks):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = self.calculate_iou(track.bbox, detection.bbox)
        
        # Improved greedy matching with better association logic
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        
        # Sort potential matches by IoU score (highest first)
        potential_matches = []
        for t in range(len(tracks)):
            for d in range(len(detections)):
                if iou_matrix[t, d] >= iou_threshold:
                    potential_matches.append((iou_matrix[t, d], t, d))
        
        # Sort by IoU score descending
        potential_matches.sort(reverse=True)
        
        # Assign matches greedily, starting with highest IoU
        for iou_score, t, d in potential_matches:
            if t in unmatched_tracks and d in unmatched_detections:
                matches.append((t, d))
                unmatched_tracks.remove(t)
                unmatched_detections.remove(d)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def update_tracks(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracks using BYTETrack algorithm
        
        Args:
            detections: List of basketball detections
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Separate high and low confidence detections
        high_conf_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_conf_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]
        
        # Predict track positions using Kalman filter
        for track in self.tracks:
            if track.kalman_filter and track.kalman_filter.initialized:
                try:
                    predicted_pos = track.kalman_filter.predict()
                    if predicted_pos is not None and len(predicted_pos) >= 2:
                        # Update bbox center based on prediction
                        cx, cy = predicted_pos[0], predicted_pos[1]
                        w = track.bbox[2] - track.bbox[0]
                        h = track.bbox[3] - track.bbox[1]
                        track.bbox = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
                except:
                    pass  # Continue with previous bbox if prediction fails
        
        # Stage 1: Associate high-confidence detections with active tracks
        active_tracks = [t for t in self.tracks if t.state == "active"]
        matches, unmatched_tracks, unmatched_high_dets = self.associate_detections_to_tracks(
            high_conf_dets, active_tracks, self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = active_tracks[track_idx]
            detection = high_conf_dets[det_idx]
            
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.hits += 1
            track.time_since_update = 0
            
            # Update Kalman filter
            if track.kalman_filter and track.kalman_filter.initialized:
                try:
                    cx = (detection.bbox[0] + detection.bbox[2]) / 2
                    cy = (detection.bbox[1] + detection.bbox[3]) / 2
                    track.kalman_filter.update(np.array([cx, cy]))
                except:
                    pass
        
        # Stage 2: Associate remaining high-confidence detections with lost tracks
        lost_tracks = [t for t in self.tracks if t.state == "lost"]
        remaining_high_dets = [high_conf_dets[i] for i in unmatched_high_dets]
        
        if lost_tracks and remaining_high_dets:
            matches2, unmatched_lost_tracks, unmatched_high_dets2 = self.associate_detections_to_tracks(
                remaining_high_dets, lost_tracks, self.match_thresh * 0.7  # More lenient for lost tracks
            )
            
            # Reactivate matched lost tracks
            for track_idx, det_idx in matches2:
                track = lost_tracks[track_idx]
                detection = remaining_high_dets[det_idx]
                
                track.bbox = detection.bbox
                track.confidence = detection.confidence
                track.state = "active"
                track.hits += 1
                track.time_since_update = 0
            
            unmatched_high_dets = [unmatched_high_dets[i] for i in unmatched_high_dets2]
        
        # Stage 3: Associate low-confidence detections with unmatched tracks
        all_unmatched_tracks = []
        for i in unmatched_tracks:
            all_unmatched_tracks.append(active_tracks[i])
        
        if all_unmatched_tracks and low_conf_dets:
            matches3, _, _ = self.associate_detections_to_tracks(
                low_conf_dets, all_unmatched_tracks, self.match_thresh * 0.4  # Even more lenient for low confidence
            )
            
            # Update matched tracks with low-confidence detections
            for track_idx, det_idx in matches3:
                track = all_unmatched_tracks[track_idx]
                detection = low_conf_dets[det_idx]
                
                track.bbox = detection.bbox
                track.confidence = detection.confidence
                track.hits += 1
                track.time_since_update = 0
        
        # Create new tracks for unmatched high-confidence detections
        for det_idx in unmatched_high_dets:
            detection = high_conf_dets[det_idx]
            new_track = Track(
                track_id=self.track_id_counter,
                bbox=detection.bbox,
                confidence=detection.confidence,
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
    
    def infer_frame(self, frame: np.ndarray) -> Tuple[List[Track], np.ndarray]:
        """
        Run inference on a single frame
        
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
            outputs = results[self.output_layer]
            logger.info(f"Inference output shape: {outputs.shape}")
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return [], frame
        
        # Postprocess detections
        detections = self.postprocess_detections(outputs, frame.shape[:2])
        logger.info(f"Found {len(detections)} detections")
        
        # Debug: print detection details
        for i, det in enumerate(detections):
            logger.info(f"Detection {i}: bbox={det.bbox}, conf={det.confidence:.3f}, class={det.class_id}")
        
        # Update tracks
        active_tracks = self.update_tracks(detections)
        logger.info(f"Active tracks: {len(active_tracks)}")
        
        # Draw clean annotations without trails
        annotated_frame = self.draw_clean_tracks(frame.copy(), active_tracks)
        
        return active_tracks, annotated_frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw track bounding boxes and IDs on frame"""
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID and confidence
            label = f"ID:{track.track_id} ({track.confidence:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def draw_clean_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw only current active tracks without trails or history"""
        # Ensure we're working with a fresh copy of the frame
        clean_frame = frame.copy()
        
        # Draw only active tracks with clean bounding boxes
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Ensure coordinates are valid
            if x2 > x1 and y2 > y1:
                # Draw clean bounding box in green
                cv2.rectangle(clean_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw track ID and confidence with background for better visibility
                label = f"Basketball ID:{track.track_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw background rectangle for text
                cv2.rectangle(clean_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(clean_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return clean_frame
    
    def draw_tracks_and_detections(self, frame: np.ndarray, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """Draw both tracks and raw detections for debugging (kept for compatibility)"""
        return self.draw_clean_tracks(frame, tracks)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get OpenVINO device information"""
        try:
            available_devices = self.core.available_devices
            device_info = {
                "available_devices": available_devices,
                "current_device": self.device.value,
                "model_path": str(self.model_path),
                "input_shape": self.input_shape
            }
            return device_info
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}


# Optimized Basketball Model class for compatibility with existing Flet UI
class OptimizedBasketballModel:
    """Wrapper class for compatibility with existing UI code"""
    
    def __init__(self, model_path: str, device: str = "CPU"):
        device_enum = DeviceType(device.upper())
        self.tracker = BasketballTracker(model_path, device_enum)
    
    def infer_frame(self, frame: np.ndarray) -> np.ndarray:
        """Infer frame and return annotated result"""
        _, annotated_frame = self.tracker.infer_frame(frame)
        return annotated_frame
    
    def get_tracks(self, frame: np.ndarray) -> List[Track]:
        """Get active tracks for a frame"""
        tracks, _ = self.tracker.infer_frame(frame)
        return tracks


if __name__ == "__main__":
    # Example usage
    model_path = "models/ov_models/basketballModel_openvino_model/basketballModel.xml"
    tracker = BasketballTracker(model_path, DeviceType.CPU)
    
    # Process video
    cap = cv2.VideoCapture("./data/video/dribbling.mov")  # or video file path
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        tracks, annotated_frame = tracker.infer_frame(frame)
        
        cv2.imshow("Basketball Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()