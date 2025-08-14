#!/usr/bin/env python3
"""
Enhanced Basketball Holding Detector

Integrates advanced pose and basketball tracking with holding detection.
Uses OpenVINO optimized models for improved performance.
"""

import cv2
import numpy as np
import time
import sys
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Add project root to path to fix imports
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import trackers
from tracker.basketballTracker import BasketballTracker
from tracker.poseTracker import PoseTracker, PoseModel
from tracker.utils.openvino_utils import DeviceType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BallHoldingDetector:
    """
    Enhanced Basketball Holding Detector
    
    Features:
    - OpenVINO optimized pose and basketball tracking
    - Advanced visualization of holding state
    - Robust detection with position tracking
    - Configurable parameters for holding detection
    """
    
    # COCO keypoint indices for wrists
    LEFT_WRIST_IDX = 9   # COCO format index
    RIGHT_WRIST_IDX = 10  # COCO format index
    
    def __init__(self, 
                 pose_model_path="models/ov_models/yolov8s-pose_openvino_model/yolov8s-pose.xml",
                 ball_model_path="models/ov_models/basketballModel_openvino_model/basketballModel.xml",
                 video_path="data/ky.mov",
                 device=DeviceType.CPU):
        """
        Initialize the ball holding detector with OpenVINO optimized trackers
        
        Args:
            pose_model_path: Path to OpenVINO pose model XML
            ball_model_path: Path to OpenVINO basketball model XML
            video_path: Path to input video
            device: OpenVINO device type (CPU, GPU, etc.)
        """
        # Initialize trackers
        try:
            self.pose_tracker = PoseTracker(pose_model_path, device)
            self.ball_tracker = BasketballTracker(ball_model_path, device)
            logger.info("Initialized OpenVINO trackers successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trackers: {e}")
            raise
        
        # Open the video source
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize holding detection parameters
        self.hold_start_time = None
        self.is_holding = False
        self.hold_duration = 1.5     # seconds 
        self.hold_threshold = 300    # pixels 
        self.hold_cooldown = 0.4     # seconds after release before detecting again
        self.last_release_time = 0
        
        # Detection confidence thresholds
        self.pose_confidence_threshold = 0.3  # Minimum confidence for pose keypoints
        self.ball_confidence_threshold = 0.5  # Minimum confidence for basketball detection
        
        # Track the last known positions
        self.last_left_wrist = None
        self.last_right_wrist = None
        self.last_ball_pos = None
        self.last_ball_bbox = None
        self.last_person_id = None   # Track which person is holding the ball
        self.frames_without_detection = 0
        self.max_frames_to_keep = 20  # Increased for better tracking during occlusions
        
        # Visualization parameters
        self.hold_indicator_size = 80
        self.hold_indicator_thickness = 4
        self.hold_indicator_color = (0, 165, 255)  # Orange
        self.hold_active_color = (0, 0, 255)       # Red
        self.hold_text_color = (255, 255, 255)     # White
        self.hold_text_bg = (0, 0, 0)              # Black
        
        # Person colors for multi-person tracking
        self.person_colors = [
            (0, 255, 255),   # Yellow
            (0, 255, 0),     # Green
            (255, 0, 0),     # Blue
            (255, 0, 255),   # Magenta
            (0, 165, 255),   # Orange
            (128, 0, 128)    # Purple
        ]
        
        # Debug mode
        self.debug = True
        
        logger.info(f"Basketball holding detector initialized with threshold: {self.hold_threshold}px")
        logger.info(f"Hold duration set to {self.hold_duration}s")
        logger.info(f"Video dimensions: {self.frame_width}x{self.frame_height} @ {self.fps}fps")

    def run(self):
        """
        Main processing loop for basketball holding detection
        """
        logger.info("Starting basketball holding detection")
        frame_count = 0
        start_time = time.time()
        
        # Process frames from the video source
        while self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                # Loop video if at the end
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            try:
                # Process the current frame
                annotated_frame, ball_detected = self.process_frame(frame)
                
                # Add FPS counter
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                           (self.frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                # Display the annotated frame
                cv2.imshow("Basketball Holding Detection", annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    # Reset detection state
                    self.reset()
                    logger.info("Detection state reset")
            
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Processed {frame_count} frames at {fps:.1f} FPS")
    
    def reset(self):
        """
        Reset the detection state
        """
        self.hold_start_time = None
        self.is_holding = False
        self.last_left_wrist = None
        self.last_right_wrist = None
        self.last_ball_pos = None
        self.last_ball_bbox = None
        self.frames_without_detection = 0

    def process_frame(self, frame):
        """Process a single frame for basketball holding detection"""
        # Track basketball
        ball_result = self._track_basketball(frame)
        ball_tracks = ball_result['tracks']
        annotated_frame = ball_result['frame']
        ball_detected, ball_x_center, ball_y_center, ball_bbox = ball_result['position']
        
        # Track poses
        pose_result = self._track_poses(frame)
        pose_detections = pose_result['detections']
        annotated_frame = pose_result['frame']
        
        if ball_detected:
            self.last_ball_pos = (ball_x_center, ball_y_center)
            self.last_ball_bbox = ball_bbox
            self.frames_without_detection = 0
            
            # Draw ball
            self._draw_ball(annotated_frame, ball_x_center, ball_y_center, ball_bbox)
        
        # Use cached ball position if no detection
        if not ball_detected and self.last_ball_pos is not None and self.frames_without_detection < self.max_frames_to_keep:
            ball_detected = True
            ball_x_center, ball_y_center = self.last_ball_pos
            ball_bbox = self.last_ball_bbox
            self.frames_without_detection += 1
            
            # Draw cached ball
            self._draw_ball(annotated_frame, ball_x_center, ball_y_center, ball_bbox, is_cached=True)
        
        # Process wrists with ball position to find closest person
        ball_position = (ball_x_center, ball_y_center) if ball_detected else None
        wrist_result = self._process_wrists(pose_detections, annotated_frame, ball_position)
        left_wrist = wrist_result['left_wrist']
        right_wrist = wrist_result['right_wrist']
        person_id = wrist_result['person_id']
        
        # Print detection status
        self._print_detection_status(ball_detected, left_wrist is not None, right_wrist is not None)
        if ball_detected and left_wrist is not None and right_wrist is not None:
            print(f"Detected - Ball: ({ball_x_center:.1f}, {ball_y_center:.1f}), Left wrist: ({left_wrist[0]:.1f}, {left_wrist[1]:.1f}), Right wrist: ({right_wrist[0]:.1f}, {right_wrist[1]:.1f})")
        
        # Process holding detection
        if ball_detected and left_wrist is not None and right_wrist is not None:
            self._process_holding_detection(annotated_frame, ball_x_center, ball_y_center, left_wrist, right_wrist)
        else:
            # Reset holding state if no ball or wrists detected
            self.is_holding = False
            self.hold_start_time = None
        
        return annotated_frame, ball_detected
        
    def _track_basketball(self, frame):
        """Track basketball in frame"""
        try:
            ball_tracks, ball_frame = self.ball_tracker.track_frame(frame)
            if ball_tracks is None or len(ball_tracks) == 0:
                return {'tracks': [], 'frame': frame.copy(), 'position': (False, 0, 0, None)}
                
            # Get highest confidence track
            sorted_tracks = self._sort_tracks_by_confidence(ball_tracks)
            if not sorted_tracks:
                return {'tracks': [], 'frame': ball_frame, 'position': (False, 0, 0, None)}
                
            # Extract ball position
            ball_track = sorted_tracks[0]
            ball_position = self._extract_ball_position(ball_track)
            
            return {'tracks': sorted_tracks, 'frame': ball_frame, 'position': ball_position}
        except Exception as e:
            logger.debug(f"Error in basketball tracking: {e}")
            return {'tracks': [], 'frame': frame.copy(), 'position': (False, 0, 0, None)}
    
    def _sort_tracks_by_confidence(self, tracks):
        """Sort tracks by confidence if available"""
        try:
            if hasattr(tracks[0], 'confidence'):
                return sorted(tracks, key=lambda x: x.confidence, reverse=True)
            return tracks
        except Exception as e:
            logger.debug(f"Error sorting tracks: {e}")
            return tracks
    
    def _extract_ball_position(self, ball_track):
        """Extract ball position from track object"""
        try:
            # Handle different track formats
            if hasattr(ball_track, 'center_x') and hasattr(ball_track, 'center_y'):
                x = ball_track.center_x
                y = ball_track.center_y
                bbox = [ball_track.x1, ball_track.y1, ball_track.x2, ball_track.y2] if all(hasattr(ball_track, attr) for attr in ['x1', 'y1', 'x2', 'y2']) else None
                return True, x, y, bbox
            elif hasattr(ball_track, 'center') and isinstance(ball_track.center, tuple) and len(ball_track.center) == 2:
                x, y = ball_track.center
                bbox = ball_track.get('bbox', None)
                return True, x, y, bbox
            elif isinstance(ball_track, dict) and 'center' in ball_track:
                x, y = ball_track['center']
                bbox = ball_track.get('bbox', None)
                return True, x, y, bbox
            else:
                logger.debug(f"Unexpected ball track format: {type(ball_track)}")
                return False, 0, 0, None
        except Exception as e:
            logger.debug(f"Error extracting ball position: {e}")
            return False, 0, 0, None
    
    def _track_poses(self, frame):
        """Track human poses in frame"""
        try:
            pose_detections, pose_frame = self.pose_tracker.infer_frame(frame)
            if pose_detections is None:
                pose_detections = []
            return {'detections': pose_detections, 'frame': pose_frame}
        except Exception as e:
            logger.debug(f"Error in pose tracking: {e}")
            return {'detections': [], 'frame': frame.copy()}
    
    def _process_wrists(self, pose_detections, frame, ball_position=None):
        """Process wrist keypoints from pose detections
        
        Args:
            pose_detections: List of pose detections
            frame: Current video frame
            ball_position: Optional tuple (x, y) of ball position for finding closest person
        """
        left_wrist = None
        right_wrist = None
        have_valid_pose = False
        person_id = None
        
        # Extract wrist positions from pose detection
        if pose_detections and len(pose_detections) > 0:
            # If we have multiple people and know ball position, find closest person to ball
            if len(pose_detections) > 1 and ball_position is not None:
                ball_x, ball_y = ball_position
                closest_person = None
                min_distance = float('inf')
                
                for i, person in enumerate(pose_detections):
                    if len(person.keypoints) <= max(self.LEFT_WRIST_IDX, self.RIGHT_WRIST_IDX):
                        continue
                        
                    # Calculate average position of both wrists
                    left_wrist_kp = person.keypoints[self.LEFT_WRIST_IDX]
                    right_wrist_kp = person.keypoints[self.RIGHT_WRIST_IDX]
                    
                    if (left_wrist_kp.confidence > self.pose_confidence_threshold and 
                        right_wrist_kp.confidence > self.pose_confidence_threshold):
                        
                        # Calculate distance from wrists to ball
                        left_dist = np.linalg.norm(np.array([left_wrist_kp.x, left_wrist_kp.y]) - 
                                                  np.array([ball_x, ball_y]))
                        right_dist = np.linalg.norm(np.array([right_wrist_kp.x, right_wrist_kp.y]) - 
                                                   np.array([ball_x, ball_y]))
                        
                        # Use minimum distance of either wrist
                        dist = min(left_dist, right_dist)
                        
                        if dist < min_distance:
                            min_distance = dist
                            closest_person = person
                            person_id = i
                
                # Use the closest person if found
                if closest_person is not None:
                    person = closest_person
                    print(f"Selected person {person_id} as closest to ball (distance: {min_distance:.1f}px)")
                else:
                    # Fall back to highest confidence person
                    person = sorted(pose_detections, key=lambda x: x.confidence, reverse=True)[0]
                    person_id = pose_detections.index(person)
                    print(f"Using highest confidence person {person_id} (no valid wrists found near ball)")
            else:
                # Use highest confidence person when only one person or no ball position
                person = sorted(pose_detections, key=lambda x: x.confidence, reverse=True)[0]
                person_id = pose_detections.index(person)
            
            if len(person.keypoints) > max(self.LEFT_WRIST_IDX, self.RIGHT_WRIST_IDX):
                left_wrist_kp = person.keypoints[self.LEFT_WRIST_IDX]
                right_wrist_kp = person.keypoints[self.RIGHT_WRIST_IDX]
                
                # Check confidence and extract coordinates
                if (left_wrist_kp.confidence > self.pose_confidence_threshold and 
                    right_wrist_kp.confidence > self.pose_confidence_threshold):
                    
                    left_wrist = np.array([left_wrist_kp.x, left_wrist_kp.y, left_wrist_kp.confidence])
                    right_wrist = np.array([right_wrist_kp.x, right_wrist_kp.y, right_wrist_kp.confidence])
                    
                    # Validate coordinates are within frame boundaries
                    if (0 <= left_wrist[0] < self.frame_width and 0 <= left_wrist[1] < self.frame_height and
                        0 <= right_wrist[0] < self.frame_width and 0 <= right_wrist[1] < self.frame_height):
                        
                        have_valid_pose = True
                        self.last_left_wrist = left_wrist
                        self.last_right_wrist = right_wrist
                        self.last_person_id = person_id
                        self.frames_without_detection = 0
                        
                        # Draw detected wrists
                        self._draw_wrist(frame, left_wrist, right_wrist, is_cached=False, person_id=person_id)
        
        # Use cached wrist positions if no valid pose detected
        if not have_valid_pose and self.last_left_wrist is not None and self.last_right_wrist is not None:
            if self.frames_without_detection < self.max_frames_to_keep:
                left_wrist = self.last_left_wrist
                right_wrist = self.last_right_wrist
                self.frames_without_detection += 1
                
                # Draw cached wrists
                self._draw_wrist(frame, left_wrist, right_wrist, is_cached=True, person_id=self.last_person_id)
            else:
                # Reset wrist tracking after too many frames without detection
                self.last_left_wrist = None
                self.last_right_wrist = None
        
        return {'left_wrist': left_wrist, 'right_wrist': right_wrist, 'person_id': person_id}
    
    def _draw_wrist(self, frame, left_wrist, right_wrist, is_cached=False, person_id=None):
        """Draw wrist markers on frame"""
        # Select color based on person ID and detection status
        if person_id is not None and not is_cached:
            color = self.person_colors[person_id % len(self.person_colors)]
        else:
            color = (0, 255, 255) if not is_cached else (0, 165, 255)  # Yellow if detected, orange if cached
        
        # Draw wrist circles
        cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 10, color, -1)
        cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 10, color, -1)
        
        # Add text labels
        status = "Cached" if is_cached else "Detected"
        person_text = f"P{person_id} " if person_id is not None else ""
        cv2.putText(frame, f"{person_text}L {status}", (int(left_wrist[0])+15, int(left_wrist[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"{person_text}R {status}", (int(right_wrist[0])+15, int(right_wrist[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_ball(self, frame, x, y, bbox=None, is_cached=False):
        """Draw ball visualization on frame
        
        Args:
            frame: Frame to draw on
            x: Ball center x-coordinate
            y: Ball center y-coordinate
            bbox: Optional bounding box [x1, y1, x2, y2]
            is_cached: Whether this is a cached ball position (for visual differentiation)
        """
        # Draw ball center with different color if cached
        color = (0, 165, 255) if is_cached else (0, 255, 0)  # Orange if cached, green otherwise
        cv2.circle(frame, (int(x), int(y)), 10, color, -1)
        
        # Draw bounding box if available
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    def _print_detection_status(self, ball_detected, left_wrist_detected, right_wrist_detected):
        """Print detection status for debugging"""
        print(f"Detection status - Ball: {ball_detected}, Left wrist: {left_wrist_detected}, Right wrist: {right_wrist_detected}")
    
    def _process_holding_detection(self, frame, ball_x, ball_y, left_wrist, right_wrist):
        """Process holding detection and visualization"""
        # Calculate distances from wrists to ball
        left_distance = np.linalg.norm(left_wrist[:2] - np.array([ball_x, ball_y]))
        right_distance = np.linalg.norm(right_wrist[:2] - np.array([ball_x, ball_y]))
        min_distance = min(left_distance, right_distance)
        
        # Calculate if ball is between hands
        is_between_hands = self._check_ball_between_hands(ball_x, ball_y, left_wrist, right_wrist)
        
        # Print distance information
        print(f"Distance to ball: {min_distance:.2f} px (threshold: {self.hold_threshold} px), Between hands: {is_between_hands}")
        
        # Draw lines from both wrists to ball
        left_line_color = (0, 255, 255) if left_distance < self.hold_threshold else (200, 200, 200)
        right_line_color = (0, 255, 255) if right_distance < self.hold_threshold else (200, 200, 200)
        cv2.line(frame, (int(left_wrist[0]), int(left_wrist[1])), (int(ball_x), int(ball_y)), left_line_color, 2)
        cv2.line(frame, (int(right_wrist[0]), int(right_wrist[1])), (int(ball_x), int(ball_y)), right_line_color, 2)
        
        # Draw line between wrists
        wrist_line_color = (0, 255, 0) if is_between_hands else (0, 0, 255)
        cv2.line(frame, (int(left_wrist[0]), int(left_wrist[1])), (int(right_wrist[0]), int(right_wrist[1])), wrist_line_color, 1)
        
        # Check holding status
        self.check_holding(left_distance, right_distance, is_between_hands)
        
        # Add visualizations
        self._draw_holding_indicator(frame, min_distance, is_between_hands)
        self._draw_info_overlay(frame, left_wrist, right_wrist, ball_x, ball_y, min_distance, is_between_hands)
    
    def _check_ball_between_hands(self, ball_x, ball_y, left_wrist, right_wrist):
        """Check if the ball is positioned between the player's hands"""
        # Create vectors
        left_to_right = np.array([right_wrist[0] - left_wrist[0], right_wrist[1] - left_wrist[1]])
        left_to_ball = np.array([ball_x - left_wrist[0], ball_y - left_wrist[1]])
        right_to_ball = np.array([ball_x - right_wrist[0], ball_y - right_wrist[1]])
        
        # Calculate distances
        wrist_distance = np.linalg.norm(left_to_right)
        
        # Project ball position onto the line between wrists
        if wrist_distance > 0:
            projection = np.dot(left_to_ball, left_to_right) / wrist_distance
            
            # Check if projection is between 0 and wrist_distance
            is_between = 0 <= projection <= wrist_distance
            
            # Check if ball is close to the line between wrists
            # Calculate perpendicular distance from ball to line between wrists
            if wrist_distance > 0:
                unit_vector = left_to_right / wrist_distance
                perpendicular_distance = abs(np.cross(unit_vector, left_to_ball))
                is_close_to_line = perpendicular_distance < self.hold_threshold / 2
            else:
                is_close_to_line = False
                
            return is_between and is_close_to_line
        else:
            # Wrists are at the same position
            return False
        
        # Add visualizations
        self._draw_holding_indicator(frame, min_distance, is_between_hands)
        self._draw_info_overlay(frame, left_wrist, right_wrist, ball_x, ball_y, min_distance, is_between_hands)
    
    def _draw_holding_indicator(self, frame, distance, is_between_hands=False):
        """
        Draw enhanced holding indicator
        """
        # Draw holding indicator in top-right corner
        indicator_x = self.frame_width - self.hold_indicator_size - 20
        indicator_y = 20
        
        # Draw indicator background
        cv2.rectangle(frame, 
                       (indicator_x - 10, indicator_y - 10),
                       (indicator_x + self.hold_indicator_size + 10, indicator_y + self.hold_indicator_size + 70),
                       (0, 0, 0), -1)
        
        # Draw holding status
        if self.is_holding:
            # Draw filled circle for active holding
            cv2.circle(frame, 
                        (indicator_x + self.hold_indicator_size//2, indicator_y + self.hold_indicator_size//2),
                        self.hold_indicator_size//2, 
                        self.hold_active_color, -1)
            
            # Add pulsing effect based on time
            pulse = int(20 * (0.5 + 0.5 * np.sin(time.time() * 5)))
            cv2.circle(frame, 
                        (indicator_x + self.hold_indicator_size//2, indicator_y + self.hold_indicator_size//2),
                        self.hold_indicator_size//2 + pulse, 
                        self.hold_active_color, 2)
            
            # Add text
            cv2.putText(frame, "HOLDING", 
                         (indicator_x, indicator_y + self.hold_indicator_size + 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.hold_text_color, 2)
        else:
            # Draw empty circle for no holding
            cv2.circle(frame, 
                        (indicator_x + self.hold_indicator_size//2, indicator_y + self.hold_indicator_size//2),
                        self.hold_indicator_size//2, 
                        self.hold_indicator_color, self.hold_indicator_thickness)
            
            # Add progress arc if potential holding is detected
            if self.hold_start_time is not None:
                hold_progress = min(1.0, (time.time() - self.hold_start_time) / self.hold_duration)
                end_angle = int(360 * hold_progress)
                
                # Draw progress arc
                center = (indicator_x + self.hold_indicator_size//2, indicator_y + self.hold_indicator_size//2)
                radius = self.hold_indicator_size//2
                cv2.ellipse(frame, center, (radius, radius), 
                           0, 0, end_angle, self.hold_active_color, self.hold_indicator_thickness)
            
            # Add distance text
            cv2.putText(frame, f"DIST: {distance:.0f}", 
                       (indicator_x, indicator_y + self.hold_indicator_size + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.hold_text_color, 2)
        
        # Add between hands status indicator
        between_color = (0, 255, 0) if is_between_hands else (0, 0, 255)
        between_text = "BETWEEN" if is_between_hands else "NOT BETWEEN"
        cv2.putText(frame, between_text, 
                    (indicator_x - 10, indicator_y + self.hold_indicator_size + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, between_color, 2)
    
    def _draw_info_overlay(self, frame, left_wrist, right_wrist, ball_x, ball_y, min_distance, is_between_hands=False):
        """
        Draw information overlay on frame
        """
        # Background for text overlay
        cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 150), (255, 255, 255), 1)
        
        # Add text information
        cv2.putText(frame, f"Ball: ({ball_x:.0f}, {ball_y:.0f})", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.hold_text_color, 1)
        
        cv2.putText(frame, f"L-Wrist: ({left_wrist[0]:.0f}, {left_wrist[1]:.0f})", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.hold_text_color, 1)
        
        cv2.putText(frame, f"R-Wrist: ({right_wrist[0]:.0f}, {right_wrist[1]:.0f})", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.hold_text_color, 1)
        
        # Distance with color based on threshold
        distance_color = (0, 255, 0) if min_distance < self.hold_threshold else (0, 165, 255)
        cv2.putText(frame, f"Distance: {min_distance:.0f} px", 
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)
        
        # Between hands status
        between_color = (0, 255, 0) if is_between_hands else (0, 0, 255)
        between_text = "Between Hands: YES" if is_between_hands else "Between Hands: NO"
        cv2.putText(frame, between_text, 
                   (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, between_color, 2)
        
        # Add threshold line to visualize the holding distance
        if self.is_holding:
            # Apply a semi-transparent overlay when holding
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), 
                         (0, 0, 200), -1)  # Blue tint
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)  # Apply overlay with 20% opacity

    def check_holding(self, left_distance, right_distance, is_between_hands):
        min_distance = min(left_distance, right_distance)
        # Always print the distance for debugging
        print(f"Distance to ball: {min_distance:.2f} px (threshold: {self.hold_threshold} px)")
        
        # Only check if wrists are close enough to ball (don't require ball between hands)
        is_close_enough = min_distance < self.hold_threshold
        is_valid_holding = is_close_enough  # Removed the is_between_hands condition
        
        # Still print the between hands status for debugging
        print(f"Distance to ball: {min_distance:.2f} px (threshold: {self.hold_threshold} px), Between hands: {is_between_hands}")
        
        if is_valid_holding:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
                print(f"Potential holding detected! Distance: {min_distance:.2f} < {self.hold_threshold}")
            elif (
                time.time() - self.hold_start_time > self.hold_duration
                and not self.is_holding
            ):
                print(f"\n*** THE BALL IS BEING HELD! *** (Distance: {min_distance:.2f})\n")
                self.is_holding = True
            elif self.is_holding:
                print(f"Still holding the ball. Distance: {min_distance:.2f}")
        else:
            if self.is_holding:
                print(f"\n*** BALL RELEASED! *** (Distance: {min_distance:.2f} > {self.hold_threshold})\n")
            self.hold_start_time = None
            self.is_holding = False


if __name__ == "__main__":
    ball_detection = BallHoldingDetector()
    ball_detection.run()