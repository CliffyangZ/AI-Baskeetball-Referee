# Avi Shah - Basketball Shot Detector/Tracker - July 2023

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from .utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device

class ShotDetector:
    def __init__(self, shared_basketball_tracker=None):
        # Load the YOLO model created from main.py - change text to your relative path
        self.overlay_text = "Waiting..."
        
        # Use shared basketball tracker if provided
        self.shared_basketball_tracker = shared_basketball_tracker
        
        # Always load hoop detection model for shot detection
        try:
            self.model = YOLO("models/pt_models/hoopModel.pt")
            # Uncomment this line to accelerate inference. Note that this may cause errors in some setups.
            #self.model.half()
            print("Loaded hoop detection model for shot detection")
        except Exception as e:
            print(f"Warning: Could not load hoop model: {e}")
            print("Shot detection will not work without hoop model")
            self.model = None
        
        if shared_basketball_tracker is not None:
            print("Using shared basketball tracker for ball tracking")
        
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        
        # Initialize variables for standalone and integrated modes
        self.cap = None
        self.standalone_mode = False
        
        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        
        # If this is called directly (not through referee.py), run in standalone mode
        if __name__ == "__main__":
            self.standalone_mode = True
            self.cap = cv2.VideoCapture("data/video/parallel_angle.mov")
            self.run()

    def run(self):
        """Run the shot detector in standalone mode with video capture."""
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break

            # Process the frame
            processed_frame = self.process_frame(self.frame)
            
            # Display the processed frame
            cv2.imshow('Frame', processed_frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        self.cap.release()
        cv2.destroyAllWindows()
        
    def process_frame(self, frame):
        """Process a single frame for shot detection.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (annotated_frame, shot_made, shot_attempted)
                - annotated_frame: Frame with shot detection visualization
                - shot_made: Boolean indicating if a shot was made in this frame
                - shot_attempted: Boolean indicating if a shot was attempted in this frame
        """
        if frame is None:
            return None, False, False
            
        # Store the frame for internal use
        self.frame = frame.copy()
        print(f"Shot counter process_frame called - Frame {self.frame_count}")
        
        # Reset shot detection flags for this frame
        shot_made = False
        shot_attempted = False
        
        # Use shared basketball tracker if available, otherwise use local model
        if self.shared_basketball_tracker is not None:
            # Get basketball coordinates from shared tracker
            basketball_tracks, _ = self.shared_basketball_tracker.track_frame(frame)
            
            # Process basketball tracks
            if basketball_tracks:
                for track_id, track_data in basketball_tracks.items():
                    # Extract position and confidence
                    center = track_data.get('center', None)
                    conf = track_data.get('confidence', 0.0)
                    
                    if center and conf > 0.3:
                        self.update_ball_position(center, track_id, conf)
                        
                        # Draw bounding box if available
                        if 'bbox' in track_data:
                            x1, y1, x2, y2 = track_data['bbox']
                            w, h = x2 - x1, y2 - y1
                            cvzone.cornerRect(self.frame, (int(x1), int(y1), int(w), int(h)))
            
            # We still need to detect hoops separately
            # This is a simplified version that assumes hoops are stationary
            # In a full implementation, you would need a separate hoop detector
            # or integrate it with the basketball tracker
        else:
            # Run object detection with local model
            results = self.model(self.frame, stream=True, device=self.device)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Only create ball points if high confidence or near hoop
                    if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                        self.update_ball_position(center, 0, conf)
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    # Create hoop points if high confidence
                    if conf > .5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

        # Process the frame for shot detection
        self.clean_motion()
        
        # Track if a shot was detected in this frame
        prev_attempts = self.attempts
        prev_makes = self.makes
        
        self.shot_detection()
        self.display_score()
        self.frame_count += 1
        
        # Check if a shot was made or attempted in this frame
        if self.attempts > prev_attempts:
            shot_attempted = True
            if self.makes > prev_makes:
                shot_made = True
        
        return self.frame, shot_made, shot_attempted

    def clean_motion(self):
        """Clean ball and hoop motion data"""
        # Clean ball motion and display current ball center
        if len(self.ball_pos) > 1:
            try:
                # Clean the ball positions - pass the current frame count
                self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
                
                # Draw the ball center safely
                try:
                    if self.ball_pos and len(self.ball_pos) > 0:
                        ball_center = self.ball_pos[-1][0]
                        if isinstance(ball_center, tuple) and len(ball_center) == 2:
                            x, y = ball_center
                            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                                cv2.circle(self.frame, (int(x), int(y)), 2, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Error drawing ball center: {e}")
            except Exception as e:
                print(f"Error in clean_ball_pos: {e}")
                # If there's an error, just keep the positions but don't try to clean them

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            try:
                # Clean the hoop positions
                self.hoop_pos = clean_hoop_pos(self.hoop_pos)
                
                # Draw the hoop center safely
                try:
                    if self.hoop_pos and len(self.hoop_pos) > 0:
                        hoop_center = self.hoop_pos[-1][0]
                        if isinstance(hoop_center, tuple) and len(hoop_center) == 2:
                            x, y = hoop_center
                            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                                cv2.circle(self.frame, (int(x), int(y)), 2, (128, 128, 0), 2)
                except Exception as e:
                    print(f"Error drawing hoop center: {e}")
            except Exception as e:
                print(f"Error in clean_hoop_pos: {e}")
                # If there's an error, just keep the positions but don't try to clean them

    def shot_detection(self):
        # Debug information
        if self.frame_count % 60 == 0:  # Print every 60 frames (about once per 2 seconds)
            print(f"Shot detection debug - Ball positions: {len(self.ball_pos)}, Hoop positions: {len(self.hoop_pos)}")
            if len(self.ball_pos) > 0:
                print(f"Latest ball position: {self.ball_pos[-1]}")
            if len(self.hoop_pos) > 0:
                print(f"Latest hoop position: {self.hoop_pos[-1]}")
        
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]
                    print(f"Ball detected in UP area at frame {self.up_frame}")

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]
                    print(f"Ball detected in DOWN area at frame {self.down_frame}")

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    print(f"SHOT ATTEMPT DETECTED! Total attempts: {self.attempts}")
                    self.up = False
                    self.down = False

                    # If it is a make, put a green overlay and display "完美"
                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        print(f"SHOT MADE! Total makes: {self.makes}")
                        self.overlay_color = (0, 255, 0)  # Green for make
                        self.overlay_text = "Make"
                        self.fade_counter = self.fade_frames

                    else:
                        self.overlay_color = (255, 0, 0)  # Red for miss
                        self.overlay_text = "Miss"
                        self.fade_counter = self.fade_frames

    def display_score(self):
        # Add text with current statistics
        text = f"{self.makes} / {self.attempts}"
        percentage = (self.makes / self.attempts * 100) if self.attempts > 0 else 0
        full_text = f"Shots: {text} ({percentage:.1f}%)"
        
        # Display shot statistics on frame
        cv2.putText(self.frame, full_text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.putText(self.frame, full_text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        # Debug: print current statistics more frequently
        print(f"Shot display called: {self.makes}/{self.attempts} ({percentage:.1f}%) - Frame {self.frame_count}")
        if self.frame_count % 30 == 0:  # Print every 30 frames
            print(f"Shot display periodic: {self.makes}/{self.attempts} ({percentage:.1f}%) - Frame {self.frame_count}")

        # Add overlay text for shot result if it exists
        if hasattr(self, 'overlay_text'):
            # Calculate text size to position it at the right top corner
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
            text_x = self.frame.shape[1] - text_width - 40  # Right alignment with some margin
            text_y = 100  # Top margin

            # Display overlay text with color (overlay_color)
            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        self.overlay_color, 6)
            # cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            try:
                # Create a colored overlay with the same shape as the frame
                overlay = np.zeros_like(self.frame)
                overlay[:] = self.overlay_color  # Fill with the color
                
                # Apply the overlay with transparency
                alpha = 0.2 * (self.fade_counter / self.fade_frames)
                self.frame = cv2.addWeighted(self.frame, 1 - alpha, overlay, alpha, 0)
                self.fade_counter -= 1
            except Exception as e:
                print(f"Error in fade effect: {e}")
                # If there's an error, just decrement the counter without applying the effect
                self.fade_counter -= 1


    def update_ball_position(self, center, track_id, confidence=1.0):
        """Update ball position from basketball tracker.
        
        Args:
            center: Tuple (x, y) of ball center coordinates
            track_id: Tracking ID of the ball (not used in current implementation)
            confidence: Detection confidence (0-1)
        """
        try:
            if center and confidence > 0.3:
                # Extract x, y values properly handling NumPy types
                try:
                    # Handle various input formats
                    if isinstance(center, tuple) and len(center) == 2:
                        x, y = center
                    elif hasattr(center, '__iter__') and len(center) == 2:
                        x, y = center[0], center[1]
                    else:
                        print(f"Unrecognized center format: {center}")
                        return
                    
                    # Convert NumPy types to Python native types if needed
                    if hasattr(x, 'item'):
                        x = x.item()
                    if hasattr(y, 'item'):
                        y = y.item()
                        
                    # Final type check and conversion
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        x, y = int(x), int(y)
                        # Estimate ball size based on typical basketball size (can be adjusted)
                        estimated_width = 30
                        estimated_height = 30
                        self.ball_pos.append(((x, y), self.frame_count, estimated_width, estimated_height, confidence))
                    else:
                        print(f"Invalid center values: x={x}, y={y} - must be numbers")
                except Exception as e:
                    print(f"Error processing center coordinates {center}: {e}")
        except Exception as e:
            print(f"Error in update_ball_position: {e}")
            # Continue without adding the position
    
    def process_frame_with_tracks(self, frame, basketball_tracks):
        """Process a frame with pre-detected basketball tracks.
        
        Args:
            frame: Input video frame
            basketball_tracks: Dictionary or list of basketball tracks from the tracker
            
        Returns:
            dict: Dictionary containing annotated frame and shot detection results
        """
        if frame is None:
            return {'annotated_frame': None}
            
        # Store the frame for internal use
        self.frame = frame.copy()
        
        # Reset shot detection flags for this frame
        shot_made = False
        shot_attempted = False
        
        # Process basketball tracks with better None checking
        if basketball_tracks is not None and basketball_tracks:
            # Handle different track formats
            if isinstance(basketball_tracks, dict):
                # Dictionary format with track_id as keys
                for track_id, track_data in basketball_tracks.items():
                    # Extract position and confidence
                    center = track_data.get('center', None)
                    conf = track_data.get('confidence', 0.0)
                    
                    if center and conf > 0.3:
                        self.update_ball_position(center, track_id, conf)
                        
                        # Draw bounding box if available
                        if 'bbox' in track_data:
                            x1, y1, x2, y2 = track_data['bbox']
                            w, h = x2 - x1, y2 - y1
                            cvzone.cornerRect(self.frame, (int(x1), int(y1), int(w), int(h)))
            elif isinstance(basketball_tracks, list):
                # List format with track objects
                for i, track_data in enumerate(basketball_tracks):
                    # Skip None or invalid track data
                    if track_data is None:
                        continue
                        
                    # Extract track ID and center
                    track_id = track_data.get('track_id', i) if isinstance(track_data, dict) else i
                    
                    # Handle different track formats
                    if isinstance(track_data, dict):
                        center = track_data.get('center', None)
                        conf = track_data.get('confidence', 0.0)
                        
                        if center and conf > 0.3:
                            self.update_ball_position(center, track_id, conf)
                            if self.frame_count % 30 == 0:  # Debug every 30 frames
                                print(f"Updated ball position: center={center}, conf={conf}")
                            
                            # Draw bounding box if available
                            if 'bbox' in track_data and track_data['bbox'] is not None:
                                try:
                                    bbox = track_data['bbox']
                                    if len(bbox) >= 4:
                                        x1, y1, x2, y2 = bbox[:4]
                                        w, h = x2 - x1, y2 - y1
                                        cvzone.cornerRect(self.frame, (int(x1), int(y1), int(w), int(h)))
                                except (ValueError, TypeError, IndexError) as e:
                                    print(f"Error processing bbox: {e}")
                    elif isinstance(track_data, (tuple, list)) and len(track_data) >= 2:
                        # Handle tuple/list format (x, y, ...)
                        try:
                            center = (track_data[0], track_data[1])
                            conf = track_data[2] if len(track_data) > 2 else 0.5
                            if center and conf > 0.3:
                                self.update_ball_position(center, track_id, conf)
                        except (IndexError, TypeError) as e:
                            print(f"Error processing track tuple: {e}")
        
        # Handle hoop detection
        if self.model is not None:
            # Run object detection for hoops only
            try:
                results = self.model(self.frame, stream=True, device=self.device)
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Bounding box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w, h = x2 - x1, y2 - y1
                            
                            # Confidence
                            conf = math.ceil((box.conf[0] * 100)) / 100
                            
                            # Class Name
                            cls = int(box.cls[0])
                            current_class = self.class_names[cls]
                            
                            center = (int(x1 + w / 2), int(y1 + h / 2))
                            
                            # Create hoop points if high confidence
                            if conf > .5 and current_class == "Basketball Hoop":
                                self.hoop_pos.append((center, self.frame_count, w, h, conf))
                                cvzone.cornerRect(self.frame, (x1, y1, w, h))
            except Exception as e:
                print(f"Error in hoop detection: {e}")
        else:
            # No hoop model available - shot detection will not work
            pass
        
        # Process the frame for shot detection
        self.clean_motion()
        
        # Track if a shot was detected in this frame
        prev_attempts = self.attempts
        prev_makes = self.makes
        
        self.shot_detection()
        self.display_score()
        self.frame_count += 1
        
        # Check if a shot was made or attempted in this frame
        if self.attempts > prev_attempts:
            shot_attempted = True
            if self.makes > prev_makes:
                shot_made = True
        
        # Ensure display_score is called before returning the frame
        self.display_score()
        
        return {
            'annotated_frame': self.frame,
            'shot_made': shot_made,
            'shot_attempted': shot_attempted
        }
        
    def get_statistics(self):
        """Get current shot statistics.
        
        Returns:
            dict: Dictionary with shot statistics
        """
        percentage = 0
        if self.attempts > 0:
            percentage = (self.makes / self.attempts) * 100
            
        return {
            "attempts": self.attempts,
            "makes": self.makes,
            "percentage": percentage
        }
    
    def reset_counter(self):
        """Reset shot statistics counters."""
        self.attempts = 0
        self.makes = 0
        self.ball_pos = []
        self.hoop_pos = []
        self.up = False
        self.down = False
        self.fade_counter = 0


if __name__ == "__main__":
    ShotDetector()
