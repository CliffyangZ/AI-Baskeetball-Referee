#!/usr/bin/env python3
"""
Basketball Shot Counter with Automatic Hoop Detection

This program detects basketball shots and counts successful baskets.
It uses:
1. The optimized basketball model to track the ball
2. YOLOv8 to detect the basketball hoop
3. Trajectory analysis to determine when a shot goes in

Usage:
    python shot_counter_auto.py [--video VIDEO_PATH] [--model MODEL_PATH] [--device DEVICE_ID]
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse
from collections import deque
from pathlib import Path
from ultralytics import YOLO

# Add the optimized_basketballModel directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "optimized_basketballModel"))

# Import from the optimized basketball model
from BasketballTracker import BasketballTracker
from PerformanceMonitor import PerformanceMonitor

class BasketballShotCounter:
    """
    Basketball Shot Counter class that detects and counts successful basketball shots.
    """
    def __init__(self, ball_model_path='../pt_models/basketballModel.pt', hoop_model_path='yolov8n.pt'):
        """
        Initialize the shot counter with basketball and hoop detection models.
        
        Args:
            ball_model_path (str): Path to the basketball YOLO model
            hoop_model_path (str): Path to the YOLOv8 model for hoop detection
        """
        # Initialize the basketball tracker
        self.ball_tracker = BasketballTracker(ball_model_path)
        
        # Initialize the hoop detector
        try:
            self.hoop_model = YOLO(hoop_model_path)
            print(f"Loaded hoop detection model: {hoop_model_path}")
        except Exception as e:
            print(f"Warning: Could not load {hoop_model_path}: {e}")
            self.hoop_model = YOLO('yolov8n.pt')
            print(f"Using YOLOv8n model instead")
        
        # Shot detection parameters
        self.ball_trajectory = deque(maxlen=30)  # Store recent ball positions
        self.hoop_positions = []  # Will store detected hoop positions
        self.hoop_position = None  # Current active hoop position (x, y, radius)
        self.hoop_confidence_threshold = 0.3  # Minimum confidence for hoop detection
        self.shot_count = 0  # Counter for successful shots
        self.last_shot_time = 0  # Time of the last detected shot
        self.shot_cooldown = 2.0  # Seconds to wait before counting another shot
        
        # Shot detection state
        self.ball_above_hoop = False
        self.ball_through_hoop = False
        self.shot_in_progress = False
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Display settings
        self.show_trajectory = True
        self.show_stats = True
        self.show_detections = True
    
    def set_hoop_position(self, x, y, radius):
        """
        Set the basketball hoop position manually.
        
        Args:
            x (int): X-coordinate of hoop center
            y (int): Y-coordinate of hoop center
            radius (int): Radius of the hoop
        """
        self.hoop_position = (x, y, radius)
        print(f"Hoop set at position ({x}, {y}) with radius {radius}")
    
    def detect_hoop(self, frame):
        """
        Detect basketball hoop in the frame using YOLOv8.
        
        Args:
            frame (numpy.ndarray): Input video frame
            
        Returns:
            tuple: (x, y, radius) of the detected hoop or None if not detected
        """
        # Use YOLOv8 to detect objects in the frame
        results = self.hoop_model.predict(
            source=frame,
            conf=self.hoop_confidence_threshold,
            verbose=False
        )
        
        # Get the first result (should only be one frame)
        result = results[0] if len(results) > 0 else None
        
        if not result or len(result.boxes) == 0:
            return None
        
        # Look for basketball hoop or rim in the detections
        hoop_detected = False
        hoop_box = None
        hoop_conf = 0
        
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.hoop_model.names[class_id].lower()
            conf = float(box.conf[0])
            
            # Check if the class is related to basketball hoop
            if ('hoop' in class_name or 'rim' in class_name or 'basket' in class_name or 'backboard' in class_name) and conf > hoop_conf:
                hoop_detected = True
                hoop_box = box
                hoop_conf = conf
        
        if hoop_detected and hoop_box is not None:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, hoop_box.xyxy[0])
            
            # Calculate center and radius
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Estimate radius based on box width
            width = x2 - x1
            height = y2 - y1
            radius = min(width, height) // 4  # Approximate hoop radius
            
            # Add to hoop positions with confidence
            hoop_pos = (center_x, center_y, radius, hoop_conf)
            self.hoop_positions.append(hoop_pos)
            
            # Keep only the last 5 detections
            if len(self.hoop_positions) > 5:
                self.hoop_positions.pop(0)
            
            # Use the average of recent detections to stabilize hoop position
            if len(self.hoop_positions) >= 3:
                avg_x = sum(pos[0] for pos in self.hoop_positions) // len(self.hoop_positions)
                avg_y = sum(pos[1] for pos in self.hoop_positions) // len(self.hoop_positions)
                avg_radius = sum(pos[2] for pos in self.hoop_positions) // len(self.hoop_positions)
                return (avg_x, avg_y, avg_radius)
            
            return (center_x, center_y, radius)
        
        return None
    
    def detect_shot(self, ball_position):
        """
        Detect if a shot has been made based on ball trajectory and hoop position.
        
        Args:
            ball_position (tuple): (x, y) position of the ball
            
        Returns:
            bool: True if a shot was made, False otherwise
        """
        if not ball_position or not self.hoop_position:
            return False
        
        ball_x, ball_y = ball_position
        hoop_x, hoop_y, hoop_radius = self.hoop_position
        
        # Calculate distance from ball to hoop center
        distance = np.sqrt((ball_x - hoop_x)**2 + (ball_y - hoop_y)**2)
        
        # Check if the ball is above the hoop
        if ball_y < hoop_y - hoop_radius and distance < hoop_radius * 3:
            self.ball_above_hoop = True
        
        # Check if the ball has passed through the hoop
        if self.ball_above_hoop and ball_y > hoop_y and distance < hoop_radius:
            self.ball_through_hoop = True
        
        # Check if the ball has completed a shot trajectory
        if self.ball_above_hoop and self.ball_through_hoop and ball_y > hoop_y + hoop_radius:
            # Reset shot detection state
            self.ball_above_hoop = False
            self.ball_through_hoop = False
            
            # Check if enough time has passed since the last shot
            current_time = time.time()
            if current_time - self.last_shot_time > self.shot_cooldown:
                self.last_shot_time = current_time
                self.shot_count += 1
                return True
        
        return False
    
    def process_frame(self, frame):
        """
        Process a video frame to detect the ball, hoop, and count shots.
        
        Args:
            frame (numpy.ndarray): Input video frame
            
        Returns:
            numpy.ndarray: Processed frame with visualizations
        """
        # Start performance monitoring
        self.performance_monitor.start_frame()
        
        # Make a copy of the frame for visualization
        processed_frame = frame.copy()
        
        # Detect basketball hoop if not already set
        if not self.hoop_position:
            hoop_pos = self.detect_hoop(frame)
            if hoop_pos:
                self.set_hoop_position(*hoop_pos)
        
        # Detect basketball in the frame
        results = self.ball_tracker.model.predict(
            source=frame,
            conf=0.25,
            iou=0.5,
            verbose=False
        )
        
        # Get the first result (should only be one frame)
        result = results[0] if len(results) > 0 else None
        
        # Initialize ball position as None
        ball_position = None
        
        # Process detection results
        if result and len(result.boxes) > 0:
            # Get the boxes and confidence scores
            boxes = result.boxes.cpu().numpy()
            
            # Find the basketball with highest confidence
            max_conf = -1
            best_box = None
            
            for box in boxes:
                # Check if the class is basketball
                class_id = int(box.cls[0])
                if class_id in self.ball_tracker.basketball_class_indices:
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                        best_box = box
            
            # If a basketball was found, get its position
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                ball_x = (x1 + x2) // 2
                ball_y = (y1 + y2) // 2
                ball_position = (ball_x, ball_y)
                
                # Add to trajectory
                self.ball_trajectory.append(ball_position)
                
                # Check if a shot was made
                if self.hoop_position:
                    shot_made = self.detect_shot(ball_position)
        
        # Visualize the results
        processed_frame = self.visualize_results(processed_frame, ball_position)
        
        # End performance monitoring
        fps = self.performance_monitor.end_frame()
        
        return processed_frame
    
    def visualize_results(self, frame, ball_position):
        """
        Visualize the detection results on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            ball_position (tuple): (x, y) position of the ball
            
        Returns:
            numpy.ndarray: Frame with visualizations
        """
        # Draw the hoop if defined
        if self.hoop_position:
            hoop_x, hoop_y, hoop_radius = self.hoop_position
            cv2.circle(frame, (hoop_x, hoop_y), hoop_radius, (0, 255, 255), 2)
            
            # Draw hoop area for shot detection
            cv2.circle(frame, (hoop_x, hoop_y), hoop_radius * 3, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Draw the ball if detected
        if ball_position:
            ball_x, ball_y = ball_position
            cv2.circle(frame, (ball_x, ball_y), 10, (0, 0, 255), -1)
        
        # Draw the ball trajectory
        if self.show_trajectory and len(self.ball_trajectory) > 1:
            for i in range(1, len(self.ball_trajectory)):
                if self.ball_trajectory[i] and self.ball_trajectory[i-1]:
                    cv2.line(
                        frame,
                        self.ball_trajectory[i-1],
                        self.ball_trajectory[i],
                        (255, 0, 0),
                        2
                    )
        
        # Draw shot count and FPS
        if self.show_stats:
            # Draw shot count
            cv2.putText(
                frame,
                f"Shots: {self.shot_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Draw FPS
            stats = self.performance_monitor.get_stats()
            fps = stats.get('avg_fps', 0)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            # Draw hoop detection status
            status = "Hoop: Detected" if self.hoop_position else "Hoop: Searching..."
            color = (0, 255, 0) if self.hoop_position else (0, 165, 255)
            cv2.putText(
                frame,
                status,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )
        
        return frame
    
    def process_video(self, video_path, output_path=None):
        """
        Process a video file to count basketball shots.
        
        Args:
            video_path (str): Path to the input video
            output_path (str, optional): Path to save the output video
            
        Returns:
            int: Number of shots counted
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return 0
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset shot counter
        self.shot_count = 0
        self.hoop_position = None
        self.hoop_positions = []
        
        print("Processing video and detecting basketball hoop automatically...")
        
        # Process the video frame by frame
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Write the frame to the output video
            if writer:
                writer.write(processed_frame)
            
            # Display the frame
            cv2.imshow("Basketball Shot Counter", processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Return the final shot count
        return self.shot_count
    
    def process_realtime(self, source=0):
        """
        Process real-time video from a camera to count basketball shots.
        
        Args:
            source (int or str): Camera index or video stream URL
            
        Returns:
            int: Number of shots counted
        """
        # Open the video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return 0
        
        # Reset shot counter
        self.shot_count = 0
        self.hoop_position = None
        self.hoop_positions = []
        
        print("Processing real-time video and detecting basketball hoop automatically...")
        
        # Process the video frame by frame
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow("Basketball Shot Counter", processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Return the final shot count
        return self.shot_count

def main():
    """
    Main function to run the basketball shot counter.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Basketball Shot Counter with Auto Hoop Detection')
    parser.add_argument('--video', type=str, default=None, help='Path to the video file')
    parser.add_argument('--ball-model', type=str, default='../pt_models/basketballModel.pt', help='Path to the basketball model')
    parser.add_argument('--hoop-model', type=str, default='yolov8n.pt', help='Path to the YOLOv8 model for hoop detection')
    parser.add_argument('--device', type=int, default=0, help='Camera device ID for real-time processing')
    args = parser.parse_args()
    
    # Create the shot counter
    shot_counter = BasketballShotCounter(
        ball_model_path=args.ball_model,
        hoop_model_path=args.hoop_model
    )
    
    # Process video or real-time feed
    if args.video:
        print(f"Processing video: {args.video}")
        output_path = f"shot_counter_output_{Path(args.video).stem}.mp4"
        shots = shot_counter.process_video(args.video, output_path)
        print(f"Video processing complete. Detected {shots} successful shots.")
        print(f"Output saved to {output_path}")
    else:
        print(f"Starting real-time shot counting from camera {args.device}")
        shots = shot_counter.process_realtime(args.device)
        print(f"Real-time processing complete. Detected {shots} successful shots.")

if __name__ == "__main__":
    main()
