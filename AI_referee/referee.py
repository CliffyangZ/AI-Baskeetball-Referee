#!/usr/bin/env python3
"""
Basketball Referee System

Integrates all AI referee components to detect violations and track statistics.
Components integrated:
- Basketball tracking (basketballTracker.py)
- Pose tracking (poseTracker.py)
- Dribble counting (dribble_counting.py)
- Ball holding detection (holding_basketball.py)
- Travel violation detection (travel_detection.py)
- Double dribble detection (double_dribble.py)
- Shot detection and counting (shot_counter.py)
- Step counting (step_counting.py)
"""

import os
import cv2
import numpy as np
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

# Add project root to path to fix imports
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import trackers
from tracker.basketballTracker import BasketballTracker
from tracker.poseTracker import PoseTracker
from tracker.utils.openvino_utils import DeviceType

# Import referee components
from AI_referee.dribble_counting import DribbleCounter
from AI_referee.holding_basketball import BallHoldingDetector
from AI_referee.travel_detection import TravelViolationDetector
from AI_referee.double_dribble import DoubleDribbleDetector
from AI_referee.shot_counter import ShotDetector
from AI_referee.step_counting import StepCounter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ViolationEvent:
    """Data class for violation events"""
    type: str  # Type of violation (travel, double_dribble, etc.)
    timestamp: float  # Time when violation was detected
    description: str  # Human-readable description
    player_id: Optional[int] = None  # ID of player who committed violation
    confidence: float = 1.0  # Confidence level of detection


@dataclass
class GameStatistics:
    """Data class for game statistics"""
    shot_attempts: int = 0
    shot_makes: int = 0
    dribble_count: int = 0
    step_count: Dict[int, int] = None  # Player ID -> step count
    violations: List[ViolationEvent] = None
    holding_duration: float = 0.0
    
    def __post_init__(self):
        if self.step_count is None:
            self.step_count = {}
        if self.violations is None:
            self.violations = []
    
    @property
    def shooting_percentage(self) -> float:
        """Calculate shooting percentage"""
        if self.shot_attempts == 0:
            return 0.0
        return (self.shot_makes / self.shot_attempts) * 100


class BasketballReferee:
    """
    Basketball Referee System
    
    Integrates all AI referee components to detect violations and track statistics.
    """
    
    def __init__(self, 
                 basketball_model_path="models/ov_models/basketballModel_openvino_model/basketballModel.xml",
                 pose_model_path="models/ov_models/yolov8s-pose_openvino_model/yolov8s-pose.xml",
                 video_path="data/video/parallel_angle.mov",
                 device=DeviceType.CPU,
                 rules="FIBA"):
        """
        Initialize the basketball referee system
        
        Args:
            basketball_model_path: Path to basketball detection model
            pose_model_path: Path to pose detection model
            video_path: Path to video file or camera index (e.g., 0)
            device: Device to run inference on (CPU, GPU, etc.)
            rules: Basketball rules to follow (FIBA, NBA)
        """
        logger.info("Initializing Basketball Referee System")
        
        # Store configuration
        self.rules = rules
        self.device = device
        
        # Initialize video source
        self.video_path = video_path
        # Support camera indices (e.g., 0/1) and keywords (camera/webcam)
        cam_index = None
        if isinstance(video_path, int):
            cam_index = video_path
        elif isinstance(video_path, str):
            vp = video_path.strip().lower()
            if vp.isdigit():
                cam_index = int(vp)
            elif vp in ("cam", "camera", "webcam"):
                cam_index = 0
        try:
            if cam_index is not None:
                logger.info(f"Opening camera index {cam_index}")
                self.cap = cv2.VideoCapture(cam_index)
            else:
                logger.info(f"Opening video file {video_path}")
                self.cap = cv2.VideoCapture(video_path)
        except Exception as e:
            logger.error(f"Failed to create VideoCapture: {e}")
            raise
        if not self.cap or not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize trackers
        try:
            self.basketball_tracker = BasketballTracker(basketball_model_path, device)
            self.pose_tracker = PoseTracker(pose_model_path, device)
            logger.info("Initialized trackers successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trackers: {e}")
            raise
        
        # Initialize referee components
        self.dribble_counter = DribbleCounter(basketball_model_path, video_path)
        self.holding_detector = BallHoldingDetector(pose_model_path, basketball_model_path, video_path, device)
        self.travel_detector = TravelViolationDetector(rules=rules)
        self.double_dribble_detector = DoubleDribbleDetector()
        self.shot_detector = ShotDetector()
        self.step_counter = StepCounter(pose_model_path, video_path)
        
        # Initialize statistics
        self.statistics = GameStatistics()
        
        # State variables
        self.frame_count = 0
        self.last_violation_time = 0
        self.violation_cooldown = 3.0  # seconds
        self.is_holding_ball = False
        
        # Visualization settings
        self.show_stats = True
        self.show_violations = True
        self.violation_display_time = 3.0  # seconds
        self.current_violations = []  # List of currently displayed violations
        
        logger.info("Basketball Referee System initialized")
    
    def reset(self):
        """Reset all detectors and statistics"""
        self.dribble_counter.reset_counter()
        self.travel_detector.reset()
        self.double_dribble_detector.reset()
        self.shot_detector.reset_counter()
        self.statistics = GameStatistics()
        self.frame_count = 0
        self.last_violation_time = 0
        self.current_violations = []
        logger.info("Reset all detectors and statistics")
    
    def process_frame(self, frame):
        """
        Process a single frame with all referee components
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with all detections and statistics
        """
        # Make a copy of the frame for annotations
        annotated_frame = frame.copy()
        
        # Track basketball
        basketball_tracks, basketball_frame = self.basketball_tracker.track_frame(frame)
        
        # Track poses
        pose_tracks, pose_frame = self.pose_tracker.infer_frame(frame)
        
        # Process dribble detection and shot detection
        for track in basketball_tracks:
            track_id = track['track_id']
            center = track['center']
            motion_info = track['motion_info']
            
            # Update dribble count
            self.dribble_counter.update_dribble_count(track_id, center, motion_info)
            
            # Update shot detector with basketball position
            self.shot_detector.update_ball_position(center, track_id)
        
        # Process holding detection
        holding_frame, is_holding = self.holding_detector.process_frame(frame.copy())
        # Update the is_holding from the detector (redundant but for clarity)
        is_holding = self.holding_detector.is_holding
        
        # Update holding state in violation detectors
        if is_holding != self.is_holding_ball:
            self.travel_detector.update_holding_state(is_holding)
            self.double_dribble_detector.update_holding_state(is_holding)
            self.is_holding_ball = is_holding
        
        # Process step counting
        step_frame, step_counts = self.step_counter.process_frame(frame.copy())
        
        # Update step counts in statistics and travel detector
        for person_id, step_count in step_counts.items():
            self.statistics.step_count[person_id] = step_count
            # Detect if a step was taken by comparing with previous count
            prev_count = self.statistics.step_count.get(person_id, 0)
            if step_count > prev_count:
                self.travel_detector.update_step_count(True)
        
        # Update dribble state in double dribble detector
        dribble_detected = self.dribble_counter.dribble_count > self.statistics.dribble_count
        if dribble_detected:
            self.double_dribble_detector.update_dribble_state(True)
            self.statistics.dribble_count = self.dribble_counter.dribble_count
            
        # Process shot detection
        try:
            shot_frame, shot_made, shot_attempted = self.shot_detector.process_frame(frame.copy())
            
            # Update shot statistics
            if shot_attempted:
                self.statistics.shot_attempts += 1
                if shot_made:
                    self.statistics.shot_makes += 1
        except Exception as e:
            logger.error(f"Error in shot detection: {e}")
            # Use original frame if shot detection fails
            shot_frame = frame.copy()
            shot_made = False
            shot_attempted = False
            
        # Ensure shot_frame is a valid numpy array with the same shape as frame
        if not isinstance(shot_frame, np.ndarray) or shot_frame.shape != frame.shape:
            logger.warning("Shot frame is invalid, using original frame instead")
            shot_frame = frame.copy()
        
        # Check for violations
        self._check_violations()
        
        # Combine all annotations onto one frame
        result_frame = self._combine_annotations(frame, basketball_frame, pose_frame, holding_frame, step_frame, shot_frame)
        
        # Add statistics and violations overlay
        if self.show_stats:
            self._add_statistics_overlay(result_frame)
        
        if self.show_violations:
            self._add_violations_overlay(result_frame)
        
        self.frame_count += 1
        return result_frame
    
    def _check_violations(self):
        """Check for violations from all detectors"""
        current_time = time.time()
        
        # Only check for new violations if cooldown period has passed
        if current_time - self.last_violation_time < self.violation_cooldown:
            return
        
        # Check travel violation
        travel_status = self.travel_detector.get_violation_status()
        if travel_status['violation_type'] == 'travel':
            violation = ViolationEvent(
                type='travel',
                timestamp=current_time,
                description=f"Travel Violation - {travel_status['step_count']} steps while holding",
                confidence=0.9
            )
            self.statistics.violations.append(violation)
            self.current_violations.append((violation, current_time))
            self.last_violation_time = current_time
            logger.info(f"Travel violation detected: {violation.description}")
        
        # Check double dribble violation
        double_dribble_status = self.double_dribble_detector.get_violation_status()
        if double_dribble_status['violation_type'] == 'double_dribble':
            violation = ViolationEvent(
                type='double_dribble',
                timestamp=current_time,
                description="Double Dribble Violation - Dribbling after holding",
                confidence=0.9
            )
            self.statistics.violations.append(violation)
            self.current_violations.append((violation, current_time))
            self.last_violation_time = current_time
            logger.info(f"Double dribble violation detected: {violation.description}")
    
    def _combine_annotations(self, original_frame, basketball_frame, pose_frame, holding_frame, step_frame, shot_frame):
        """Combine annotations from all components onto one frame"""
        # Start with the original frame to preserve brightness
        result = original_frame.copy()
        
        # Extract basketball tracking annotations (only the annotations, not the background)
        try:
            basketball_mask = cv2.absdiff(basketball_frame, original_frame)
            basketball_mask = cv2.cvtColor(basketball_mask, cv2.COLOR_BGR2GRAY)
            _, basketball_mask = cv2.threshold(basketball_mask, 25, 255, cv2.THRESH_BINARY)
            basketball_mask = cv2.cvtColor(basketball_mask, cv2.COLOR_GRAY2BGR)
            basketball_elements = cv2.bitwise_and(basketball_frame, basketball_mask)
            result = cv2.addWeighted(result, 1.0, basketball_elements, 1.0, 0)
        except Exception as e:
            logger.error(f"Error applying basketball annotations: {e}")
        
        # Add pose keypoints and skeletons (without bounding boxes to avoid clutter)
        try:
            pose_mask = cv2.absdiff(pose_frame, original_frame)
            pose_mask = cv2.cvtColor(pose_mask, cv2.COLOR_BGR2GRAY)
            _, pose_mask = cv2.threshold(pose_mask, 25, 255, cv2.THRESH_BINARY)
            pose_mask = cv2.cvtColor(pose_mask, cv2.COLOR_GRAY2BGR)
            pose_elements = cv2.bitwise_and(pose_frame, pose_mask)
            result = cv2.addWeighted(result, 1.0, pose_elements, 1.0, 0)
        except Exception as e:
            logger.error(f"Error applying pose annotations: {e}")
        
        # Add holding indicators
        try:
            # Ensure frames are compatible for absdiff
            if holding_frame.shape == original_frame.shape and holding_frame.dtype == original_frame.dtype:
                holding_mask = cv2.absdiff(holding_frame, original_frame)
                holding_mask = cv2.cvtColor(holding_mask, cv2.COLOR_BGR2GRAY)
                _, holding_mask = cv2.threshold(holding_mask, 25, 255, cv2.THRESH_BINARY)
                holding_mask = cv2.cvtColor(holding_mask, cv2.COLOR_GRAY2BGR)
                holding_elements = cv2.bitwise_and(holding_frame, holding_mask)
                result = cv2.addWeighted(result, 1.0, holding_elements, 1.0, 0)
            else:
                logger.warning("Holding frame and original frame are not compatible for absdiff")
        except Exception as e:
            logger.error(f"Error applying holding indicators: {e}")
        
        # Add step counting indicators
        try:
            # Ensure frames are compatible for absdiff
            if step_frame.shape == original_frame.shape and step_frame.dtype == original_frame.dtype:
                step_mask = cv2.absdiff(step_frame, original_frame)
                step_mask = cv2.cvtColor(step_mask, cv2.COLOR_BGR2GRAY)
                _, step_mask = cv2.threshold(step_mask, 25, 255, cv2.THRESH_BINARY)
                step_mask = cv2.cvtColor(step_mask, cv2.COLOR_GRAY2BGR)
                step_elements = cv2.bitwise_and(step_frame, step_mask)
                result = cv2.addWeighted(result, 1.0, step_elements, 1.0, 0)
            else:
                logger.warning("Step frame and original frame are not compatible for absdiff")
        except Exception as e:
            logger.error(f"Error applying step indicators: {e}")
        
        # Add shot detection indicators
        try:
            # Ensure frames are compatible for absdiff
            if shot_frame.shape == original_frame.shape and shot_frame.dtype == original_frame.dtype:
                shot_mask = cv2.absdiff(shot_frame, original_frame)
                shot_mask = cv2.cvtColor(shot_mask, cv2.COLOR_BGR2GRAY)
                _, shot_mask = cv2.threshold(shot_mask, 25, 255, cv2.THRESH_BINARY)
                shot_mask = cv2.cvtColor(shot_mask, cv2.COLOR_GRAY2BGR)
                shot_elements = cv2.bitwise_and(shot_frame, shot_mask)
                result = cv2.addWeighted(result, 1.0, shot_elements, 1.0, 0)
            else:
                logger.warning("Shot frame and original frame are not compatible for absdiff")
        except Exception as e:
            logger.error(f"Error applying shot indicators: {e}")
        
        return result
    
    def _add_statistics_overlay(self, frame):
        """Add statistics overlay to the frame"""
        # Create a semi-transparent panel for statistics
        panel_height = 180
        panel_width = 300
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:, :] = (30, 30, 30)  # Dark gray background
        
        # Add statistics text
        cv2.putText(panel, "GAME STATISTICS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(panel, f"Shots: {self.statistics.shot_makes}/{self.statistics.shot_attempts}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        shooting_pct = self.statistics.shooting_percentage
        cv2.putText(panel, f"Shooting: {shooting_pct:.1f}%", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(panel, f"Dribbles: {self.statistics.dribble_count}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        total_steps = sum(self.statistics.step_count.values())
        cv2.putText(panel, f"Steps: {total_steps}", 
                   (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        violations_count = len(self.statistics.violations)
        cv2.putText(panel, f"Violations: {violations_count}", 
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add panel to the frame with transparency
        x_offset = 20
        y_offset = 20
        alpha = 0.7
        
        # Get the region of interest in the frame
        roi = frame[y_offset:y_offset+panel_height, x_offset:x_offset+panel_width]
        
        # Blend the panel with the ROI
        blended_roi = cv2.addWeighted(roi, 1-alpha, panel, alpha, 0)
        
        # Put the blended ROI back into the frame
        frame[y_offset:y_offset+panel_height, x_offset:x_offset+panel_width] = blended_roi
    
    def _add_violations_overlay(self, frame):
        """Add violations overlay to the frame"""
        current_time = time.time()
        
        # Remove expired violations from display
        self.current_violations = [(v, t) for v, t in self.current_violations 
                                  if current_time - t < self.violation_display_time]
        
        if not self.current_violations:
            return
        
        # Display the most recent violations
        y_pos = 100
        for violation, timestamp in self.current_violations:
            # Calculate fade effect based on time
            time_diff = current_time - timestamp
            alpha = 1.0 - (time_diff / self.violation_display_time)
            
            # Create a semi-transparent red bar for violation
            bar_height = 40
            bar_width = frame.shape[1] - 100
            bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
            
            if violation.type == 'travel':
                bar_color = (0, 0, 255)  # Red for travel
            elif violation.type == 'double_dribble':
                bar_color = (0, 165, 255)  # Orange for double dribble
            else:
                bar_color = (255, 0, 0)  # Blue for other violations
            
            bar[:, :] = bar_color
            
            # Add violation text
            cv2.putText(bar, f"{violation.type.upper()}: {violation.description}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add bar to the frame with transparency
            x_offset = 50
            alpha_bar = alpha * 0.8
            
            # Get the region of interest in the frame
            roi = frame[y_pos:y_pos+bar_height, x_offset:x_offset+bar_width]
            
            # Blend the bar with the ROI
            blended_roi = cv2.addWeighted(roi, 1-alpha_bar, bar, alpha_bar, 0)
            
            # Put the blended ROI back into the frame
            frame[y_pos:y_pos+bar_height, x_offset:x_offset+bar_width] = blended_roi
            
            y_pos += bar_height + 10
    
    def get_statistics(self) -> GameStatistics:
        """Get current game statistics"""
        return self.statistics
    
    def get_violations(self) -> List[ViolationEvent]:
        """Get list of detected violations"""
        return self.statistics.violations
    
    def run(self):
        """Run the basketball referee system on the video"""
        logger.info("Starting Basketball Referee System")
        logger.info("Press 'q' to quit, 'r' to reset, 's' to toggle statistics display")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                logger.info("End of video or error reading frame")
                break
            
            try:
                # Process the frame with all components
                result_frame = self.process_frame(frame)
                
                # Display the result
                cv2.imshow("Basketball Referee System", result_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested quit")
                    break
                elif key == ord('r'):
                    logger.info("User requested reset")
                    self.reset()
                elif key == ord('s'):
                    self.show_stats = not self.show_stats
                    logger.info(f"Statistics display {'enabled' if self.show_stats else 'disabled'}")
                elif key == ord('v'):
                    self.show_violations = not self.show_violations
                    logger.info(f"Violations display {'enabled' if self.show_violations else 'disabled'}")
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Basketball Referee System stopped")


def main():
    """Main entry point"""
    try:
        import argparse
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='AI Basketball Referee System')
        parser.add_argument('--video', type=str, default=0,
                          help='Path to input video file')
        parser.add_argument('--basketball-model', type=str, 
                          default="models/ov_models/basketballModel_openvino_model/basketballModel.xml",
                          help='Path to basketball detection model')
        parser.add_argument('--pose-model', type=str,
                          default="models/ov_models/yolov8s-pose_openvino_model/yolov8s-pose.xml",
                          help='Path to pose detection model')
        args = parser.parse_args()
        
        # Configuration
        basketball_model = args.basketball_model
        pose_model = args.pose_model
        video_path = args.video
        
        # Check if models exist
        if not os.path.exists(basketball_model):
            logger.error(f"Basketball model not found: {basketball_model}")
            return
        
        if not os.path.exists(pose_model):
            logger.error(f"Pose model not found: {pose_model}")
            return
        
        # Initialize and run referee system
        referee = BasketballReferee(
            basketball_model_path=basketball_model,
            pose_model_path=pose_model,
            video_path=video_path,
            device=DeviceType.CPU,
            rules="FIBA"
        )
        
        referee.run()
        
        # Print final statistics
        stats = referee.get_statistics()
        print("\n===== FINAL GAME STATISTICS =====")
        print(f"Shots: {stats.shot_makes}/{stats.shot_attempts} ({stats.shooting_percentage:.1f}%)")
        print(f"Dribbles: {stats.dribble_count}")
        print(f"Total Steps: {sum(stats.step_count.values())}")
        print(f"Violations: {len(stats.violations)}")
        for i, v in enumerate(stats.violations):
            print(f"  {i+1}. {v.type.upper()}: {v.description}")
        
    except Exception as e:
        logger.error(f"Error running Basketball Referee System: {e}")


if __name__ == "__main__":
    main()