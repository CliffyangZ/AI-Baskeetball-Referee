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
import sys
import cv2
import time
import logging
import numpy as np
import concurrent.futures
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path

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
    
    def update_dribble_count(self, count: int) -> None:
        """Update the dribble count"""
        self.dribble_count = count
        
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
                 video_path=None,
                 device="AUTO",
                 rules="FIBA"):
        """
        Initialize the basketball referee system
        
        Args:
            basketball_model_path: Path to basketball detection model
            pose_model_path: Path to pose detection model
            video_path: Path to video file or camera index (e.g., 0)
            device: Device to run inference on (AUTO, CPU, GPU)
            rules: Basketball rules to follow (FIBA, NBA)
        """
        logger.info("Initializing Basketball Referee System")
        
        # Store configuration
        self.rules = rules
        
        # Convert string device to DeviceType enum
        if isinstance(device, str):
            device_str = device.upper()
            if device_str == "CPU":
                self.device = DeviceType.CPU
            elif device_str == "GPU" or device_str == "CUDA":
                self.device = DeviceType.GPU
            elif device_str == "NPU":
                self.device = DeviceType.NPU
            else:
                self.device = DeviceType.AUTO
        else:
            # Already a DeviceType enum
            self.device = device
            
        logger.info(f"Using device: {self.device} for inference")
        
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
        
        # Initialize trackers (shared instances)
        try:
            # Pass the device as a string value instead of enum
            device_str = self.device.value if hasattr(self.device, 'value') else str(self.device)
            self.basketball_tracker = BasketballTracker(basketball_model_path, device_str)
            self.pose_tracker = PoseTracker(pose_model_path, device_str)
            logger.info("Initialized shared tracker instances successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trackers: {e}")
            raise
        
        # Initialize referee components with shared trackers
        self.dribble_counter = DribbleCounter(shared_basketball_tracker=self.basketball_tracker, 
                                           video_path=video_path)
        self.holding_detector = BallHoldingDetector(shared_pose_tracker=self.pose_tracker,
                                                 shared_basketball_tracker=self.basketball_tracker,
                                                 video_path=video_path, 
                                                 device=device)
        self.step_counter = StepCounter(shared_pose_tracker=self.pose_tracker,
                                      model_path=pose_model_path, 
                                      video_path=video_path)
        self.shot_detector = ShotDetector(shared_basketball_tracker=self.basketball_tracker)
        # Initialize travel detector with the same rules setting
        self.travel_detector = TravelViolationDetector(rules=self.rules)
        # Initialize double dribble detector
        self.double_dribble_detector = DoubleDribbleDetector()
        
        # Initialize statistics
        self.statistics = GameStatistics()
        
        # State variables
        self.frame_count = 0
        self.last_violation_time = 0
        self.violation_cooldown = 3.0  # seconds
        self.is_holding_ball = False
        
        # Initialize thread pool for asynchronous processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.async_results = {}
        
        # Configure which components can run asynchronously
        self.use_async = True  # Can be toggled to disable async processing
        
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
        Process a single frame with all basketball referee components
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with all detections and violations
        """
        # Create a copy of the frame for annotation
        result_frame = frame.copy()
        
        # Track basketball and poses once (shared instances) - critical operations, run every frame
        # These are critical and should not be run asynchronously
        basketball_tracks, basketball_frame = self.basketball_tracker.track_frame(frame)
        pose_tracks, pose_frame = self.pose_tracker.infer_frame(frame)
        
        # Constants for frame skipping
        DRIBBLE_FRAME_INTERVAL = 2  # Process every 2nd frame
        HOLDING_FRAME_INTERVAL = 1  # Process every frame (critical)
        STEP_FRAME_INTERVAL = 2     # Process every 2nd frame
        SHOT_FRAME_INTERVAL = 3     # Process every 3rd frame
        TRAVEL_FRAME_INTERVAL = 2   # Process every 2nd frame
        ASYNC_TIMEOUT = 0.1         # Timeout for async operations in seconds
        
        # Initialize default values for skipped frames
        dribble_frame = frame.copy()
        dribble_count = self.dribble_counter.dribble_count if hasattr(self.dribble_counter, 'dribble_count') else 0
        holding_frame = frame.copy()
        is_holding = self.is_holding_ball
        step_frame = frame.copy()
        step_counts = {}
        shot_frame = frame.copy()
        shot_made = False
        shot_attempted = False
        
        # Helper function for asynchronous processing
        def process_component(component_name, frame_interval, process_func, frame):
            if self.frame_count % frame_interval == 0:
                return process_func(frame)
            return None
        
        # Submit asynchronous tasks if enabled
        futures = {}
        if self.use_async:
            # Process dribble counting asynchronously with frame skipping
            if self.frame_count % DRIBBLE_FRAME_INTERVAL == 0:
                # Create a function to update dribble count using basketball tracks
                def update_dribble_count(tracks, frame):
                    for track in tracks:
                        track_id = track['track_id']
                        center = track['center']
                        motion_info = track['motion_info'] if 'motion_info' in track else {}
                        self.dribble_counter.update_dribble_count(track_id, center, motion_info)
                    return frame.copy(), self.dribble_counter.dribble_count
                
                # Submit the task with basketball_tracks we already have
                futures['dribble'] = self.thread_pool.submit(
                    update_dribble_count, basketball_tracks, frame)
            
            # Process step counting asynchronously with frame skipping
            if self.frame_count % STEP_FRAME_INTERVAL == 0:
                futures['step'] = self.thread_pool.submit(
                    self.step_counter.process_frame, frame)
            
            # Process shot detection asynchronously with frame skipping
            if self.frame_count % SHOT_FRAME_INTERVAL == 0:
                futures['shot'] = self.thread_pool.submit(
                    self.shot_detector.process_frame, frame)
        
        # Process ball holding detection (critical for rules, no skipping, run synchronously)
        holding_frame, is_holding = self.holding_detector.process_frame(frame)
        self.is_holding_ball = is_holding
        
        # Collect results from asynchronous tasks
        if self.use_async:
            # Get dribble results if available
            if 'dribble' in futures:
                try:
                    tracks_info, dribble_frame = futures['dribble'].result(timeout=ASYNC_TIMEOUT)
                    if dribble_frame is not None:
                        # Get the dribble count from the counter object
                        dribble_count = self.dribble_counter.dribble_count
                        self.statistics.update_dribble_count(dribble_count)
                except (concurrent.futures.TimeoutError, Exception) as e:
                    if isinstance(e, concurrent.futures.TimeoutError):
                        logger.warning("Dribble detection timed out, using previous results")
                    else:
                        logger.error(f"Error in dribble detection: {e}")
            
            # Get step counting results if available
            if 'step' in futures:
                try:
                    step_frame, step_counts = futures['step'].result(timeout=ASYNC_TIMEOUT)
                except (concurrent.futures.TimeoutError, Exception) as e:
                    if isinstance(e, concurrent.futures.TimeoutError):
                        logger.warning("Step counting timed out, using previous results")
                    else:
                        logger.error(f"Error in step counting: {e}")
            
            # Get shot detection results if available
            if 'shot' in futures:
                try:
                    shot_result = futures['shot'].result(timeout=ASYNC_TIMEOUT)
                    # The process_frame method returns (annotated_frame, shot_made, shot_attempted)
                    if isinstance(shot_result, tuple) and len(shot_result) == 3:
                        shot_frame, shot_made, shot_attempted = shot_result
                        
                        # Update shot statistics
                        if shot_attempted:
                            self.statistics.shot_attempts += 1
                            if shot_made:
                                self.statistics.shot_makes += 1
                                
                        # Ensure shot_frame is a valid numpy array with the same shape as frame
                        if not isinstance(shot_frame, np.ndarray) or shot_frame.shape != frame.shape:
                            logger.warning("Shot frame is invalid, using original frame instead")
                            shot_frame = frame.copy()
                except (concurrent.futures.TimeoutError, Exception) as e:
                    if isinstance(e, concurrent.futures.TimeoutError):
                        logger.warning("Shot detection timed out, using previous results")
                    else:
                        logger.error(f"Error in shot detection: {e}")
        else:
            # Synchronous processing for components when async is disabled
            # Process dribble counting with frame skipping
            if self.frame_count % DRIBBLE_FRAME_INTERVAL == 0:
                # Use basketball tracker's results directly
                # We already have basketball_tracks from earlier in this method
                dribble_frame = frame.copy()
                
                # Update dribble count for each tracked basketball
                for track in basketball_tracks:
                    track_id = track['track_id']
                    center = track['center']
                    motion_info = track['motion_info'] if 'motion_info' in track else {}
                    self.dribble_counter.update_dribble_count(track_id, center, motion_info)
                    
                dribble_count = self.dribble_counter.dribble_count
                
                # Process step counting with frame skipping
                if self.frame_count % STEP_FRAME_INTERVAL == 0:
                    step_frame, step_counts = self.step_counter.process_frame(frame)
            
            # Process shot detection with frame skipping
            if self.frame_count % SHOT_FRAME_INTERVAL == 0:
                try:
                    shot_frame, shot_made, shot_attempted = self.shot_detector.process_frame(frame)
                    
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
        
        # Update dribble state if needed
        dribble_detected = self.dribble_counter.dribble_count > self.statistics.dribble_count
        if dribble_detected:
            self.double_dribble_detector.update_dribble_state(True)
            self.statistics.dribble_count = self.dribble_counter.dribble_count
        
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
        """Combine annotations from all components onto one frame using optimized processing"""
        # Start with the original frame to preserve brightness
        result = original_frame.copy()
        
        # Create a common function for extracting annotations to avoid code duplication
        def extract_annotations(annotated_frame, original):
            if annotated_frame is None or original is None:
                return None
                
            if annotated_frame.shape != original.shape or annotated_frame.dtype != original.dtype:
                return None
                
            try:
                # Use a more efficient approach with fewer conversions
                # Calculate difference between annotated frame and original
                diff = cv2.absdiff(annotated_frame, original)
                
                # Convert to grayscale and threshold in one step
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
                
                # Create a 3-channel mask more efficiently
                mask_3ch = cv2.merge([mask, mask, mask])
                
                # Extract only the annotations
                return cv2.bitwise_and(annotated_frame, mask_3ch)
            except Exception as e:
                logger.error(f"Error extracting annotations: {e}")
                return None
        
        # Process all frames in a batch to reduce redundant operations
        frames_to_process = [
            ("basketball", basketball_frame),
            ("pose", pose_frame),
            ("holding", holding_frame),
            ("step", step_frame),
            ("shot", shot_frame)
        ]
        
        # Extract and combine annotations in one pass
        for name, frame in frames_to_process:
            annotations = extract_annotations(frame, original_frame)
            if annotations is not None:
                # Add annotations to result frame
                result = cv2.addWeighted(result, 1.0, annotations, 1.0, 0)
            else:
                logger.debug(f"No annotations extracted for {name} frame")
        
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
        parser.add_argument('--video', type=str, default='data/video/parallel_angle.mov',
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