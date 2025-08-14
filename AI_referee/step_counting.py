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
    def __init__(self, model_path="models/ov_models/yolov8s-pose_openvino_model/yolov8s-pose.xml", 
                 video_path="data/ky.mov"):
        """
        Initialize the step counter with PoseTracker using OpenVINO
        
        Args:
            model_path: Path to the OpenVINO IR model (.xml file)
            video_path: Path to the video file to analyze
        """
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # Load the PoseTracker model
        logger.info(f"Loading PoseTracker with OpenVINO model from {model_path}")
        self.pose_tracker = PoseTracker(model_path, DeviceType.CPU)
        
        # Open the video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            raise ValueError(f"Cannot open video: {video_path}")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Define the body part indices for YOLOv8 pose
        self.body_index = {"left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}
        
        # Initialize step counting variables
        self.step_count = 0
        self.prev_left_ankle_y = None
        self.prev_right_ankle_y = None
        self.step_threshold = 12
        self.min_wait_frames = 8
        self.wait_frames = 0
        
        logger.info("Step counter initialized")
    
    def reset_counter(self):
        """
        Reset the step counter and related variables
        """
        self.step_count = 0
        self.prev_left_ankle_y = None
        self.prev_right_ankle_y = None
        self.wait_frames = 0
        logger.info("Step counter reset")
    
    def process_frame(self, frame):
        """
        Process a single frame to detect steps
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated_frame, step_count)
        """
        # Run pose detection using PoseTracker
        pose_detections, annotated_frame = self.pose_tracker.infer_frame(frame)
        
        # Process keypoints if any person is detected
        if pose_detections and len(pose_detections) > 0:
            # Get keypoints for the first person detected
            pose = pose_detections[0]
            keypoints = pose.keypoints
            
            # Extract ankle and knee keypoints
            try:
                left_knee = keypoints[self.body_index["left_knee"]]
                right_knee = keypoints[self.body_index["right_knee"]]
                left_ankle = keypoints[self.body_index["left_ankle"]]
                right_ankle = keypoints[self.body_index["right_ankle"]]
                
                # Check if all keypoints have sufficient confidence
                if (left_knee.confidence > 0.5 and right_knee.confidence > 0.5 and 
                    left_ankle.confidence > 0.5 and right_ankle.confidence > 0.5):
                    
                    # Check for step if we have previous positions and not in cooldown
                    if (self.prev_left_ankle_y is not None and 
                        self.prev_right_ankle_y is not None and 
                        self.wait_frames == 0):
                        
                        # Calculate vertical movement of ankles
                        left_diff = abs(left_ankle.y - self.prev_left_ankle_y)
                        right_diff = abs(right_ankle.y - self.prev_right_ankle_y)
                        
                        # If either ankle moved more than threshold, count as step
                        if max(left_diff, right_diff) > self.step_threshold:
                            self.step_count += 1
                            logger.info(f"Step detected! Count: {self.step_count}")
                            
                            # Draw step count on frame
                            cv2.putText(annotated_frame, f"Steps: {self.step_count}", 
                                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Set cooldown to prevent multiple detections
                            self.wait_frames = self.min_wait_frames
                    
                    # Update previous positions
                    self.prev_left_ankle_y = left_ankle.y
                    self.prev_right_ankle_y = right_ankle.y
                    
                    # Decrease cooldown counter if active
                    if self.wait_frames > 0:
                        self.wait_frames -= 1
                        
            except Exception as e:
                logger.error(f"Error processing keypoints: {e}")
        
        # Always show step count on frame
        cv2.putText(annotated_frame, f"Steps: {self.step_count}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame, self.step_count
    
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