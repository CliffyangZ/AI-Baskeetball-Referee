
#!/usr/bin/env python3
"""
Basketball Dribble Counter

This module implements a basketball dribble counter using the BasketballTracker
to detect and track the ball, and count dribbles based on vertical motion.
"""

import cv2
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import basketball tracker
from tracker.basketballTracker import BasketballTracker
from tracker.utils.openvino_utils import DeviceType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DribbleCounter:
    def __init__(self, shared_basketball_tracker=None, model_path="models/ov_models/basketballModel_openvino_model/basketballModel.xml", 
                 video_path="data/video/dribbling.mov", 
                 config_path="tracker/basketball_config.yaml"):
        """
        Initialize the dribble counter with BasketballTracker
        
        Args:
            shared_basketball_tracker: Optional shared BasketballTracker instance
            model_path: Path to the OpenVINO basketball detection model (used only if shared_tracker is None)
            video_path: Path to the video file to analyze
            config_path: Path to the basketball tracker configuration file (used only if shared_tracker is None)
        """
        # Use shared tracker if provided, otherwise create a new one
        if shared_basketball_tracker is not None:
            self.tracker = shared_basketball_tracker
            logger.info("Using shared BasketballTracker instance")
        else:
            # Initialize a new basketball tracker if no shared instance provided
            self.tracker = BasketballTracker(model_path, DeviceType.CPU, config_path)
            logger.info("Created new BasketballTracker instance")
        
        # Open the video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            raise ValueError(f"Cannot open video: {video_path}")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize dribble detection variables
        self.dribble_count = 0
        self.prev_y_center = None
        self.prev_delta_y = 0
        self.dribble_threshold = 70  # Increased threshold to avoid false positives
        self.cooldown_frames = 0     # Cooldown counter to avoid multiple detections for same bounce
        self.cooldown_period = 5     # Number of frames to wait before detecting another dribble
        
        # For visualization
        self.dribble_positions = []  # Store positions where dribbles were detected
        self.max_dribble_markers = 10  # Maximum number of markers to show
        
        logger.info("Dribble counter initialized with single-ball tracking")

    def run(self):
        """
        Run the dribble counter on the video
        """
        frame_count = 0
        logger.info("Starting dribble counter")
        logger.info("Press 'q' to quit, 'r' to reset counter")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            try:
                # Track basketballs in the frame
                tracks_info, annotated_frame = self.tracker.track_frame(frame)
                
                # Process each tracked basketball
                for track in tracks_info:
                    track_id = track['track_id']
                    center = track['center']
                    motion_info = track['motion_info']
                    
                    # Update dribble count for this track
                    self.update_dribble_count(track_id, center, motion_info)
                
                # Add dribble count to the frame
                self.draw_dribble_info(annotated_frame)
                
                # Display
                cv2.imshow("Basketball Dribble Counter", annotated_frame)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                continue
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset counter
                self.reset_counter()
                logger.info("Dribble counter reset")
        
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Dribble counter stopped")

    def update_dribble_count(self, track_id, center, motion_info):
        """
        Update dribble count based on ball vertical motion
        
        A dribble is counted when the ball changes direction from moving down to moving up
        with sufficient velocity change.
        
        Args:
            track_id: ID of the tracked basketball (always 0 with our modified tracker)
            center: (x, y) coordinates of the basketball
            motion_info: Dictionary with motion information from the tracker
        """
        x, y = center
        velocity = motion_info.get('velocity', (0, 0))
        vy = velocity[1]  # Vertical velocity component
        
        # Since we're only tracking one ball (ID 0), we can simplify our tracking
        if self.prev_y_center is None:
            self.prev_y_center = y
            self.prev_delta_y = 0
            return
        
        # Decrease cooldown counter if active
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            
        # Calculate vertical movement
        delta_y = y - self.prev_y_center
        
        # Check for direction change (from down to up)
        # This happens when the ball bounces off the floor
        if (self.cooldown_frames == 0 and  # Only detect if not in cooldown
            self.prev_delta_y > self.dribble_threshold and 
            delta_y < -self.dribble_threshold):
            
            # Count as dribble and store position for visualization
            self.dribble_count += 1
            self.dribble_positions.append((center, 20))  # Store position with fade counter
            
            # Limit the number of markers
            if len(self.dribble_positions) > self.max_dribble_markers:
                self.dribble_positions.pop(0)
                
            # Set cooldown to avoid multiple detections for same bounce
            self.cooldown_frames = self.cooldown_period
                
            logger.info(f"Dribble detected! Count: {self.dribble_count}")
        
        # Update previous values
        self.prev_y_center = y
        self.prev_delta_y = delta_y
    def draw_dribble_info(self, frame):
        """
        Draw dribble count and markers on the frame
        """
        # Draw dribble count
        count_text = f"Dribbles: {self.dribble_count}"
        cv2.putText(frame, count_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (255, 255, 255), 4)
        cv2.putText(frame, count_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (0, 0, 255), 2)
        
        # Draw dribble markers with fade effect
        for i, ((x, y), fade_counter) in enumerate(self.dribble_positions):
            if fade_counter > 0:
                # Calculate alpha based on fade counter
                alpha = fade_counter / 20.0
                color = (0, int(255 * alpha), 0)  # Green with fade
                
                # Draw circle at dribble position
                cv2.circle(frame, (int(x), int(y)), 15, color, -1)
                cv2.circle(frame, (int(x), int(y)), 15, (255, 255, 255), 2)
                
                # Add number label
                label = str(self.dribble_count - (len(self.dribble_positions) - 1 - i))
                cv2.putText(frame, label, (int(x) - 5, int(y) + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Decrease fade counter
                self.dribble_positions[i] = ((x, y), fade_counter - 1)

    def reset_counter(self):
        """Reset the dribble counter"""
        self.dribble_count = 0
        self.prev_y_center = None
        self.prev_delta_y = 0
        self.cooldown_frames = 0
        self.dribble_positions = []
        
    def get_dribble_count(self):
        """Get the current dribble count
        
        Returns:
            int: Current dribble count
        """
        return self.dribble_count

    def process_frame(self, frame):
        """Process a single frame for dribble counting
        
        Args:
            frame: Video frame to process
            
        Returns:
            Tuple of (result_dict, annotated_frame)
        """
        try:
            # Track basketballs in the frame
            tracks_info, annotated_frame = self.tracker.track_frame(frame)
            
            # Process each tracked basketball
            for track in tracks_info:
                track_id = track['track_id']
                center = track['center']
                motion_info = track['motion_info'] if 'motion_info' in track else {}
                
                # Update dribble count for this track
                self.update_dribble_count(track_id, center, motion_info)
            
            # Add dribble count to the frame
            self.draw_dribble_info(annotated_frame)
            
            # Return result
            result = {
                'dribble_count': self.dribble_count
            }
            
            return result, annotated_frame
            
        except Exception as e:
            logger.error(f"Error in dribble counter: {e}")
            return {'dribble_count': self.dribble_count}, frame.copy()
            
    def process_frame_with_tracks(self, frame, basketball_tracks):
        """Process a frame with pre-detected basketball tracks.
        
        Args:
            frame: Input video frame
            basketball_tracks: Dictionary or list of basketball tracks from the tracker
            
        Returns:
            Tuple of (result_dict, annotated_frame)
        """
        try:
            # Create a copy of the frame for annotation
            annotated_frame = frame.copy()
            
            # Validate input parameters
            if frame is None:
                logger.warning("Frame is None in process_frame_with_tracks")
                return {'dribble_count': self.dribble_count}, annotated_frame
            
            if basketball_tracks is None:
                logger.debug("Basketball tracks is None, skipping track processing")
                self.draw_dribble_info(annotated_frame)
                return {'dribble_count': self.dribble_count}, annotated_frame
            
            # Process each tracked basketball
            if basketball_tracks:
                # Handle different track formats
                if isinstance(basketball_tracks, dict):
                    # Dictionary format with track_id as keys
                    for track_id, track_data in basketball_tracks.items():
                        # Extract position and motion information
                        center = track_data.get('center', None)
                        motion_info = track_data.get('motion_info', {})
                        
                        if center:
                            # Update dribble count for this track
                            self.update_dribble_count(track_id, center, motion_info)
                            
                            # Draw bounding box if available
                            if 'bbox' in track_data:
                                x1, y1, x2, y2 = track_data['bbox']
                                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                elif isinstance(basketball_tracks, list) or isinstance(basketball_tracks, tuple):
                    # List or tuple format with track objects
                    for i, track_data in enumerate(basketball_tracks):
                        # Skip None or invalid track data
                        if track_data is None:
                            continue
                            
                        # Extract track ID and center
                        track_id = i
                        center = None
                        motion_info = {}
                        
                        # Handle different track data formats
                        if isinstance(track_data, dict):
                            track_id = track_data.get('track_id', i)
                            center = track_data.get('center', None)
                            motion_info = track_data.get('motion_info', {})
                            
                            if center:
                                # Update dribble count for this track
                                self.update_dribble_count(track_id, center, motion_info)
                                
                                # Draw bounding box if available
                                if 'bbox' in track_data:
                                    try:
                                        x1, y1, x2, y2 = track_data['bbox']
                                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                    except (ValueError, TypeError) as e:
                                        logger.debug(f"Error processing bbox: {e}")
                        elif isinstance(track_data, tuple) and len(track_data) >= 2:
                            # Assume tuple is (x, y) coordinates
                            try:
                                center = (track_data[0], track_data[1])
                                
                                if center:
                                    # Update dribble count for this track
                                    self.update_dribble_count(track_id, center, motion_info)
                            except (IndexError, TypeError) as e:
                                logger.debug(f"Error processing track tuple: {e}")
            
            # Add dribble count to the frame
            self.draw_dribble_info(annotated_frame)
            
            # Return result
            result = {
                'dribble_count': self.dribble_count
            }
            
            return result, annotated_frame
            
        except Exception as e:
            logger.error(f"Error in dribble counter with tracks: {e}")
            return {'dribble_count': self.dribble_count}, frame.copy()


def main():
    """
    Main function to run the dribble counter
    """
    # Configuration
    model_path = "models/ov_models/basketballModel_openvino_model/basketballModel.xml"
    video_path = "data/video/dribbling.mov"
    config_path = "tracker/basketball_config.yaml"
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Initialize and run dribble counter
    try:
        dribble_counter = DribbleCounter(model_path, video_path, config_path)
        dribble_counter.run()
    except Exception as e:
        logger.error(f"Failed to run dribble counter: {e}")


if __name__ == "__main__":
    main()