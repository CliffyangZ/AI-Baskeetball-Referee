import cv2
import numpy as np
import logging
import os
import sys
from pathlib import Path

# Add project root to path to fix imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PoseTracker
from tracker.poseTracker import PoseTracker, DeviceType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StepCounter:
    def __init__(self, shared_pose_tracker=None, model_path="models/ov_models/yolov8s-pose_openvino_model/yolov8s-pose.xml", 
                 video_path="data/ky.mov"):
        """
        Initialize the step counter with PoseTracker using OpenVINO
        
        Args:
            shared_pose_tracker: Optional shared PoseTracker instance
            model_path: Path to the OpenVINO IR model (.xml file) (used only if shared_pose_tracker is None)
            video_path: Path to the video file to analyze
        """
        # Use shared pose tracker if provided, otherwise create a new one
        if shared_pose_tracker is not None:
            self.pose_tracker = shared_pose_tracker
            logger.info("Using shared PoseTracker instance")
        else:
            # Check if model exists
            if not os.path.exists(model_path):
                logger.error(f"Model not found: {model_path}")
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Initialize PoseTracker
            logger.info(f"Loading PoseTracker with OpenVINO model from {model_path}")
            self.pose_tracker = PoseTracker(model_path, DeviceType.CPU)
        
        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            raise IOError(f"Could not open video file: {video_path}")
        
        # Step counting parameters
        self.base_step_threshold = 15  # Base threshold for step detection
        self.min_wait_frames = 10  # Minimum frames to wait between steps
        self.distance_adaptation = True  # Enable distance-based threshold adaptation
        
        # Dictionary to store data for each person
        self.person_data = {}
        
        # Body keypoint indices
        self.body_index = {
            "nose": 0,
            "left_eye": 1,
            "right_eye": 2,
            "left_ear": 3,
            "right_ear": 4,
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
            "left_hip": 11,
            "right_hip": 12,
            "left_knee": 13,
            "right_knee": 14,
            "left_ankle": 15,
            "right_ankle": 16
        }
        
        logger.info("Step counter initialized")
    
    def reset_counter(self):
        """
        Reset the step counter and related variables for all persons
        """
        self.person_data = {}
        logger.info("Step counter reset for all persons")
    
    def process_frame(self, frame):
        """
        Process a single frame to detect steps
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated_frame, step_count)
        """
        # Run pose detection using PoseTracker but get raw detections only
        pose_detections, _ = self.pose_tracker.infer_frame(frame)
        
        # Create a clean copy of the frame for our custom visualization
        annotated_frame = frame.copy()
        
        # Draw pose skeletons and bounding boxes with our custom style
        for i, pose in enumerate(pose_detections):
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in pose.bbox]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            # Draw keypoints and skeleton
            for j, keypoint in enumerate(pose.keypoints):
                if keypoint.visible and keypoint.confidence > 0.3:
                    x, y = int(keypoint.x), int(keypoint.y)
                    # Draw keypoint
                    cv2.circle(annotated_frame, (x, y), 4, (0, 255, 255), -1)
            
            # Draw skeleton connections
            skeleton_connections = [
                (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 6), (5, 11), (6, 12),         # Shoulders to hips
                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]
            
            for connection in skeleton_connections:
                idx1, idx2 = connection
                if (idx1 < len(pose.keypoints) and idx2 < len(pose.keypoints) and
                    pose.keypoints[idx1].visible and pose.keypoints[idx2].visible and
                    pose.keypoints[idx1].confidence > 0.3 and pose.keypoints[idx2].confidence > 0.3):
                    
                    pt1 = (int(pose.keypoints[idx1].x), int(pose.keypoints[idx1].y))
                    pt2 = (int(pose.keypoints[idx2].x), int(pose.keypoints[idx2].y))
                    cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)
        
        # Process keypoints for all detected persons
        if pose_detections and len(pose_detections) > 0:
            # Process each detected person
            for pose in pose_detections:
                person_id = pose.person_id
                keypoints = pose.keypoints
                
                # Initialize person data if not exists
                if person_id not in self.person_data:
                    self.person_data[person_id] = {
                        'step_count': 0,
                        'prev_left_ankle_y': None,
                        'prev_right_ankle_y': None,
                        'wait_frames': 0,
                        'current_threshold': self.base_step_threshold,
                        'person_height': None,
                        'height_history': [],
                        'left_diff': 0,
                        'right_diff': 0,
                        'max_diff': 0,
                        'step_detected': False
                    }
                    logger.info(f"New person detected with ID: {person_id}")
            
                # Extract ankle and knee keypoints
                try:
                    left_knee = keypoints[self.body_index["left_knee"]]
                    right_knee = keypoints[self.body_index["right_knee"]]
                    left_ankle = keypoints[self.body_index["left_ankle"]]
                    right_ankle = keypoints[self.body_index["right_ankle"]]
                    
                    # Get person's tracking data
                    person_tracking = self.person_data[person_id]
                    
                    # Check if all keypoints have sufficient confidence
                    if (left_knee.confidence > 0.5 and right_knee.confidence > 0.5 and 
                        left_ankle.confidence > 0.5 and right_ankle.confidence > 0.5):
                        
                        # Check for step if we have previous positions and not in cooldown
                        if (person_tracking['prev_left_ankle_y'] is not None and 
                            person_tracking['prev_right_ankle_y'] is not None and 
                            person_tracking['wait_frames'] == 0):
                            
                            # Calculate vertical movement of ankles
                            left_diff = abs(left_ankle.y - person_tracking['prev_left_ankle_y'])
                            right_diff = abs(right_ankle.y - person_tracking['prev_right_ankle_y'])
                            
                            # Store the ankle movement for visualization
                            person_tracking['left_diff'] = left_diff
                            person_tracking['right_diff'] = right_diff
                            person_tracking['max_diff'] = max(left_diff, right_diff)
                            
                            # Calculate adaptive threshold based on person's height in the frame
                            # This helps adjust for distance - people further away appear smaller
                            threshold = self.base_step_threshold
                            
                            if self.distance_adaptation:
                                # Calculate person height (distance between ankle and shoulder)
                                left_shoulder = keypoints[self.body_index["left_shoulder"]]
                                right_shoulder = keypoints[self.body_index["right_shoulder"]]
                                
                                # Get bounding box coordinates for fallback
                                x1, y1, x2, y2 = map(int, pose.bbox)
                                
                                if (left_shoulder.confidence > 0.5 and left_ankle.confidence > 0.5):
                                    person_height = abs(left_shoulder.y - left_ankle.y)
                                elif (right_shoulder.confidence > 0.5 and right_ankle.confidence > 0.5):
                                    person_height = abs(right_shoulder.y - right_ankle.y)
                                else:
                                    # Fallback if shoulders aren't visible
                                    person_height = abs(y2 - y1)  # Use bounding box height
                                
                                # Store the person height for tracking
                                person_tracking['person_height'] = person_height
                                
                                # Add to height history for smoothing (keep last 5 values)
                                person_tracking['height_history'].append(person_height)
                                if len(person_tracking['height_history']) > 5:
                                    person_tracking['height_history'] = person_tracking['height_history'][-5:]
                                
                                # Use average height for more stable threshold
                                avg_height = sum(person_tracking['height_history']) / len(person_tracking['height_history'])
                                
                                # Scale threshold proportionally with height (smaller person = smaller threshold)
                                # Reference height of 400 pixels (close to camera)
                                reference_height = 400
                                height_ratio = avg_height / reference_height
                                threshold = max(5, self.base_step_threshold * height_ratio)
                                
                                # Store the current adaptive threshold for debugging/display
                                person_tracking['current_threshold'] = threshold
                            
                            # If either ankle moved more than adaptive threshold, count as step
                            if max(left_diff, right_diff) > threshold:
                                person_tracking['step_count'] += 1
                                logger.info(f"Step detected for Person {person_id}! Count: {person_tracking['step_count']}")
                                
                                # Set cooldown to prevent multiple detections
                                person_tracking['wait_frames'] = self.min_wait_frames
                                
                                # Mark this as a detected step for visualization
                                person_tracking['step_detected'] = True
                            else:
                                person_tracking['step_detected'] = False
                        
                        # Update previous positions
                        person_tracking['prev_left_ankle_y'] = left_ankle.y
                        person_tracking['prev_right_ankle_y'] = right_ankle.y
                        
                        # Decrease cooldown counter if active
                        if person_tracking['wait_frames'] > 0:
                            person_tracking['wait_frames'] -= 1
                            
                        # We'll draw the step counts in a separate section below
                        
                except Exception as e:
                    logger.error(f"Error processing keypoints: {e}")
        
        # Create a clean UI panel for step counting display
        # Create semi-transparent overlay for the stats panel
        h, w = annotated_frame.shape[:2]
        panel_height = 30 + (len(self.person_data) * 40) + 40  # Header + person rows (increased height) + padding
        panel_width = 280  # Increased width to accommodate visualization
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        # Draw panel border
        cv2.rectangle(annotated_frame, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
        
        # Draw title
        cv2.putText(annotated_frame, "STEP COUNTER", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw horizontal separator
        cv2.line(annotated_frame, (20, 45), (panel_width-10, 45), (255, 255, 255), 1)
        
        # Show individual step counts with adaptive threshold info
        sorted_persons = sorted(self.person_data.items())
        for i, (person_id, data) in enumerate(sorted_persons):
            y_pos = 70 + (i * 40)  # Increased vertical spacing between persons
            
            # Person ID with colored circle
            cv2.circle(annotated_frame, (30, y_pos-5), 8, (0, 165, 255), -1)
            cv2.putText(annotated_frame, f"{person_id}", 
                       (27, y_pos-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Step count
            cv2.putText(annotated_frame, f"Steps: {data['step_count']}", 
                       (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show adaptive threshold and movement visualization if enabled
            if self.distance_adaptation and 'current_threshold' in data:
                # Display threshold value
                threshold_text = f"Threshold: {data['current_threshold']:.1f}"
                cv2.putText(annotated_frame, threshold_text, 
                          (160, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                
                # Draw movement meter to visualize ankle movement vs threshold
                meter_x = 160
                meter_y = y_pos + 10
                meter_width = 80
                meter_height = 6
                
                # Background bar
                cv2.rectangle(annotated_frame, (meter_x, meter_y), 
                             (meter_x + meter_width, meter_y + meter_height), 
                             (100, 100, 100), -1)
                
                # Threshold marker
                threshold_x = meter_x + int((data['current_threshold'] / 30) * meter_width)
                cv2.line(annotated_frame, 
                         (threshold_x, meter_y - 2), 
                         (threshold_x, meter_y + meter_height + 2), 
                         (255, 200, 0), 2)
                
                # Current movement level
                movement_width = int((min(data['max_diff'], 30) / 30) * meter_width)
                movement_color = (0, 255, 0) if data['step_detected'] else (0, 165, 255)
                cv2.rectangle(annotated_frame, (meter_x, meter_y), 
                             (meter_x + movement_width, meter_y + meter_height), 
                             movement_color, -1)
        
        # Show total steps at the bottom of the panel
        total_steps = sum(person['step_count'] for person in self.person_data.values())
        y_pos = panel_height - 15
        cv2.line(annotated_frame, (20, y_pos-20), (panel_width-10, y_pos-20), (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Total Steps: {total_steps}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add instructions at the bottom of the frame
        cv2.putText(annotated_frame, "Press 'q' to quit, 'r' to reset", 
                   (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Return annotated frame and dictionary of person step counts
        person_steps = {person_id: data['step_count'] for person_id, data in self.person_data.items()}
        return annotated_frame, person_steps
    
    def run(self):
        """
        Main processing loop for step counting
        """
        logger.info("Starting step counter")
        logger.info("Press 'q' to quit, 'r' to reset counter")
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                logger.info("End of video or error reading frame")
                break
            
            # Process the frame
            annotated_frame, _ = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow("Step Counter", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User requested quit")
                break
            elif key == ord('r'):
                logger.info("User requested reset")
                self.reset_counter()
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Step counter stopped")


def main():
    """
    Main entry point
    """
    try:
        step_counter = StepCounter()
        step_counter.run()
    except Exception as e:
        logger.error(f"Error running step counter: {e}")


if __name__ == "__main__":
    main()