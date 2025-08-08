from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import yaml
import os

from BasketballTracker import BasketballTracker
from KalmanFilter import BasketballKalmanFilter


class EnhancedBasketballTracker(BasketballTracker):
    """
    Enhanced basketball tracker with advanced features like motion prediction
    and multi-stage association.
    """
    def __init__(self, model_path='basketballModel.pt'):
        """
        Initialize the enhanced basketball tracker.
        
        Args:
            model_path (str): Path to the YOLO model file
        """
        super().__init__(model_path)
        
        # Enhanced settings
        self.use_kalman_filter = True
        self.use_motion_prediction = True
        self.kalman_filters = {}
        
    def _process_and_visualize_results(self, frame, result):
        """
        Process tracking results with enhanced features and visualize them.
        
        Args:
            frame (numpy.ndarray): Input frame
            result (ultralytics.engine.results.Results): Tracking results
            
        Returns:
            numpy.ndarray: Processed frame with visualizations
        """
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        try:
            # Check if we have valid tracking results
            if result is None:
                return vis_frame
                
            # Check if boxes and IDs exist using hasattr to avoid attribute errors
            if not hasattr(result, 'boxes') or result.boxes is None:
                return vis_frame
                
            # Check if we have tracking IDs
            if not hasattr(result.boxes, 'id') or result.boxes.id is None:
                return vis_frame
                
            # Extract tracking information safely
            boxes = result.boxes.xywh.cpu().numpy() if hasattr(result.boxes, 'xywh') else []
            track_ids = result.boxes.id.int().cpu().tolist() if hasattr(result.boxes, 'id') else []
            confidences = result.boxes.conf.float().cpu().tolist() if hasattr(result.boxes, 'conf') else []
            
            # Process each detected basketball
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                # Extract box coordinates
                x_center, y_center, width, height = box
                
                # Apply Kalman filtering if enabled
                if self.use_kalman_filter:
                    # Initialize Kalman filter for new tracks
                    if track_id not in self.kalman_filters:
                        self.kalman_filters[track_id] = BasketballKalmanFilter()
                        self.kalman_filters[track_id].initialize(np.array([x_center, y_center, 0, 0]))
                    
                    # Update Kalman filter with new measurement
                    filtered_state = self.kalman_filters[track_id].update(np.array([x_center, y_center]))
                    
                    # Use filtered position
                    x_center, y_center = filtered_state[0], filtered_state[1]
                
                # Calculate bounding box coordinates
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                # Generate a consistent color for this track ID
                color = self._get_color_by_id(track_id)
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label if enabled
                if self.show_labels:
                    label = f"ID:{track_id} {conf:.2f}"
                    cv2.putText(
                        vis_frame, 
                        label, 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2
                    )
                
                # Update trajectory
                if self.show_trajectories:
                    center_point = (int(x_center), int(y_center))
                    if track_id not in self.trajectories:
                        self.trajectories[track_id] = deque(maxlen=self.max_trajectory_points)
                    self.trajectories[track_id].append(center_point)
                    
                    # Draw trajectory
                    self._draw_trajectory(vis_frame, self.trajectories[track_id], color)
                    
                    # Draw predicted trajectory if enabled
                    if self.use_motion_prediction and len(self.trajectories[track_id]) >= 3:
                        self._draw_predicted_trajectory(vis_frame, self.trajectories[track_id], color)
        except Exception as e:
            print(f"Warning: Error processing enhanced tracking results - {e}")
            # Return the original frame if processing fails
        
        return vis_frame
    
    def _draw_predicted_trajectory(self, frame, trajectory, color, num_predictions=5):
        """
        Draw predicted future trajectory based on current motion.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            trajectory (deque): Queue of trajectory points
            color (tuple): RGB color for the trajectory
            num_predictions (int): Number of future points to predict
        """
        if len(trajectory) < 3:
            return
            
        # Get the last few points
        points = list(trajectory)[-3:]
        
        # Calculate velocity vector (simple linear prediction)
        vx = points[2][0] - points[1][0]
        vy = points[2][1] - points[1][1]
        
        # Draw predicted points
        last_point = points[2]
        for i in range(1, num_predictions + 1):
            # Predict next position
            next_x = int(last_point[0] + vx * i)
            next_y = int(last_point[1] + vy * i)
            
            # Draw predicted point
            cv2.circle(frame, (next_x, next_y), 2, color, -1)
            
            # Connect to previous point
            if i == 1:
                cv2.line(frame, last_point, (next_x, next_y), color, 1, cv2.LINE_DASHED)
            else:
                prev_x = int(last_point[0] + vx * (i-1))
                prev_y = int(last_point[1] + vy * (i-1))
                cv2.line(frame, (prev_x, prev_y), (next_x, next_y), color, 1, cv2.LINE_DASHED)
