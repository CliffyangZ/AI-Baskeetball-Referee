#!/usr/bin/env python3
"""
Enhanced Basketball Tracker Demo
Demonstrates improved tracking with physics-aware Kalman filter and occlusion handling
"""

from re import S
import cv2
import numpy as np
import yaml
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Any
import sys

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use relative imports instead of package imports
from tracker.utils.EnhancedKalmanFilter import EnhancedBasketballKalmanFilter
from tracker.utils.openvino_utils import DeviceType, OpenVINOInferenceEngine
from tracker.utils.byte_track import BasketballDetection
from tracker.utils.algorithm import calculate_iou, hungarian_matching, greedy_matching, apply_nms

class BasketballTracker:
    """
    basketball tracker with improved occlusion handling
    """
    
    def __init__(self, model_path: str, device: DeviceType = DeviceType.CPU, 
                 config_path: str = "basketball_config.yaml"):
        """Initialize enhanced tracker"""
        self.model_path = model_path
        self.device = device
        
        # Load configuration
        self.config = self._load_config(config_path)
       
        # Initialize inference engine
        self.inference_engine = OpenVINOInferenceEngine(model_path, device)
        
        # Initialize enhanced Kalman filters for each track
        self.kalman_filters = {}
        self.track_id_counter = 0
        
        # Tracking parameters from config
        self.high_thresh = self.config['detection']['high_thresh']
        self.low_thresh = self.config['detection']['low_thresh']
        self.max_occlusion_frames = self.config['occlusion']['max_prediction_frames']
        
        logger.info(f"Enhanced basketball tracker initialized")
        logger.info(f"Occlusion handling: {self.max_occlusion_frames} frames")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'detection': {'high_thresh': 0.8, 'low_thresh': 0.1, 'nms_thresh': 0.5},
            'tracking': {'track_buffer': 30, 'match_thresh': 0.8},
            'kalman': {'dt': 1.0, 'gravity': 2.0, 'max_occlusion_frames': 15},
            'occlusion': {'max_prediction_frames': 15, 'confidence_decay': 0.95}
        }

    def detect_basketballs(self, frame: np.ndarray) -> List[BasketballDetection]:
        """Detect basketballs in frame"""
        # Preprocess frame
        input_tensor = self.inference_engine.preprocess_frame(frame)
        
        # Run inference
        outputs = self.inference_engine.infer(input_tensor)
        
        # Postprocess detections
        detections = []
        frame_height, frame_width = frame.shape[:2]
        
        # Handle different output formats
        if len(outputs.shape) == 3:
            outputs = outputs[0]
        
        if outputs.shape[0] < outputs.shape[1] and outputs.shape[0] <= 85:
            outputs = outputs.T
        
        # Process each detection
        for detection in outputs:
            if len(detection) >= 5:
                x_center, y_center, width, height, confidence = detection[:5]
                
                # Filter by low threshold
                if confidence >= self.low_thresh:
                    # Convert to pixel coordinates
                    input_size = (self.inference_engine.input_width, self.inference_engine.input_height)
                    frame_size = (frame_width, frame_height)
                    
                    # Simple coordinate conversion (you can use your normalize_coordinates function)
                    x1 = max(0, min((x_center - width/2) * frame_width / input_size[0], frame_width))
                    y1 = max(0, min((y_center - height/2) * frame_height / input_size[1], frame_height))
                    x2 = max(0, min((x_center + width/2) * frame_width / input_size[0], frame_width))
                    y2 = max(0, min((y_center + height/2) * frame_height / input_size[1], frame_height))
                    
                    basketball_detection = BasketballDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(confidence)
                    )
                    detections.append(basketball_detection)

        # Apply Non-Maximum Suppression to remove duplicate detections
        detections = apply_nms(detections, iou_threshold=self.config['detection']['nms_thresh'])
        
        return detections
    
    def track_frame(self, frame: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        Track basketballs in frame with enhanced occlusion handling
        
        Returns:
            Tuple of (tracks_info, annotated_frame)
        """
        # Detect basketballs
        detections = self.detect_basketballs(frame)
        
        # Enhanced tracking logic
        tracks_info = []
        annotated_frame = frame.copy()
        
        # Simple association for demo (you can integrate with ByteTrack)
        for detection in detections:
            center = detection.center
            confidence = detection.confidence
            
            # Find closest existing track or create new one
            track_id = self._associate_detection(center, confidence)
            
            # Update Kalman filter
            if track_id in self.kalman_filters:
                kf = self.kalman_filters[track_id]
                kf.update(np.array(center), confidence)
            else:
                # Create new track
                kf = EnhancedBasketballKalmanFilter(
                    dt=self.config['kalman']['dt'],
                    gravity=self.config['kalman']['gravity']
                )
                kf.initialize(np.array(center))
                self.kalman_filters[track_id] = kf
            
            # Get motion info
            motion_info = kf.get_motion_info()
            
            tracks_info.append({
                'track_id': track_id,
                'bbox': detection.bbox,
                'center': center,
                'confidence': confidence,
                'motion_info': motion_info
            })
            
            # Draw on frame
            self._draw_track(annotated_frame, detection, track_id, motion_info)
        
        # Predict for occluded tracks
        self._predict_occluded_tracks(annotated_frame, tracks_info)
        
        return tracks_info, annotated_frame
    
    def _associate_detection(self, center: Tuple[float, float], confidence: float) -> int:
        """Simple detection association (can be improved with Hungarian algorithm)"""
        min_distance = float('inf')
        best_track_id = None
        
        for track_id, kf in self.kalman_filters.items():
            if kf.can_predict_during_occlusion():
                predicted_pos = kf.get_position()
                if predicted_pos:
                    distance = np.linalg.norm(np.array(center) - np.array(predicted_pos))
                    if distance < min_distance and distance < 50:  # Distance threshold
                        min_distance = distance
                        best_track_id = track_id
        
        if best_track_id is None:
            # Create new track
            self.track_id_counter += 1
            best_track_id = self.track_id_counter
        
        return best_track_id
    
    def _predict_occluded_tracks(self, frame: np.ndarray, active_tracks: List[dict]):
        """Predict positions for occluded tracks"""
        active_track_ids = {track['track_id'] for track in active_tracks}
        
        for track_id, kf in list(self.kalman_filters.items()):
            if track_id not in active_track_ids:
                # This track is occluded, try to predict
                if kf.can_predict_during_occlusion():
                    predicted_state = kf.predict()
                    if predicted_state is not None:
                        predicted_pos = (predicted_state[0], predicted_state[1])
                        motion_info = kf.get_motion_info()
                        
                        # Draw predicted position
                        #self._draw_predicted_track(frame, predicted_pos, track_id, motion_info)
                else:
                    # Remove track if too long without detection
                    logger.debug(f"Removing track {track_id} after long occlusion")
                    del self.kalman_filters[track_id]
    
    def _draw_track(self, frame: np.ndarray, detection: BasketballDetection, 
                   track_id: int, motion_info: dict):
        """Draw active track"""
        x1, y1, x2, y2 = map(int, detection.bbox)
        center = detection.center
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
        
        # Draw track ID and info
        label = f"Ball {track_id}"
        if motion_info.get('motion_phase'):
            label += f" ({motion_info['motion_phase']})"
        
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw velocity vector if available
        velocity = motion_info.get('velocity')
        if velocity and np.linalg.norm(velocity) > 1:
            end_point = (
                int(center[0] + velocity[0] * 3),
                int(center[1] + velocity[1] * 3)
            )
            cv2.arrowedLine(frame, (int(center[0]), int(center[1])), end_point, (255, 0, 0), 2)
    
    def _draw_predicted_track(self, frame: np.ndarray, predicted_pos: Tuple[float, float], 
                            track_id: int, motion_info: dict):
        """Draw predicted track during occlusion"""
        x, y = int(predicted_pos[0]), int(predicted_pos[1])
        
        # Draw predicted position with different color
        cv2.circle(frame, (x, y), 8, (0, 0, 255), 2)  # Red circle for prediction
        cv2.putText(frame, f"Ball {track_id} (predicted)", (x+10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw uncertainty ellipse based on occlusion frames
        occlusion_frames = motion_info.get('occlusion_frames', 0)
        uncertainty_radius = min(20, 5 + occlusion_frames * 2)
        cv2.circle(frame, (x, y), uncertainty_radius, (0, 0, 255), 1)


def main():
    """Demo the enhanced basketball tracker"""
    # Configuration
    model_path = "models/ov_models/basketballModel_openvino_model/basketballModel.xml"
    video_path = "./data/ky.mov"
    
    # Check if files exist
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Initialize tracker
    try:
        tracker = BasketballTracker(model_path, DeviceType.CPU)
    except Exception as e:
        logger.error(f"Failed to initialize tracker: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
    
    logger.info("Enhanced Basketball Tracker Demo")
    logger.info("Features:")
    logger.info("- Physics-aware Kalman filtering")
    logger.info("- Occlusion prediction (red circles)")
    logger.info("- Motion phase detection")
    logger.info("- Velocity visualization (blue arrows)")
    logger.info("Press 'q' to quit, 'r' to reset tracker")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        try:
            # Track frame
            tracks_info, annotated_frame = tracker.track_frame(frame)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Active tracks: {len(tracks_info)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display
            cv2.imshow("Enhanced Basketball Tracker", annotated_frame)
            
            # Print tracking info
            if frame_count % 30 == 0:  # Every 30 frames
                logger.info(f"Frame {frame_count}: {len(tracks_info)} active tracks")
                for track in tracks_info:
                    motion = track['motion_info']
                    logger.info(f"  Track {track['track_id']}: {motion['motion_phase']}, "
                              f"speed: {motion['speed']:.1f}, occlusion: {motion['occlusion_frames']}")
            
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            continue
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset tracker
            tracker.kalman_filters.clear()
            tracker.track_id_counter = 0
            logger.info("Tracker reset")
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Demo completed")


if __name__ == "__main__":
    main()
