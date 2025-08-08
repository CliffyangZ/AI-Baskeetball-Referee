from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import yaml
import os

from PerformanceMonitor import PerformanceMonitor

class BasketballTracker:
    """
    Basketball tracking class that integrates YOLO detection with ByteTrack
    for optimized basketball tracking performance.
    """
    def __init__(self, model_path='basketballModel.pt'):
        """
        Initialize the basketball tracker with the specified model.
        
        Args:
            model_path (str): Path to the YOLO model file
        """
        try:
            # Try to load the specified model
            self.model = YOLO(model_path)
            print(f"Loaded model: {model_path}")
            model_type = "custom"
        except Exception as e:
            print(f"Warning: Could not load {model_path}: {e}")
            print("Falling back to YOLOv8n model...")
            # Fall back to a default model if the specified one doesn't exist
            self.model = YOLO('yolov8n.pt')
            print(f"Using YOLOv8n model instead")
            model_type = "default"
        
        # Print model information
        if hasattr(self.model, 'names'):
            print(f"Model classes: {self.model.names}")
        """
        # Test the model on a simple image to verify it works
        try:
            # Create a test image (black background)
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            # Draw a white circle (simulating a ball)
            cv2.circle(test_img, (320, 320), 50, (255, 255, 255), -1)
            
            # Run inference on test image
            test_results = self.model(test_img, verbose=False)
            if test_results and len(test_results) > 0 and hasattr(test_results[0], 'boxes'):
                boxes = test_results[0].boxes
                if len(boxes) > 0:
                    print(f"Model test successful! Detected {len(boxes)} objects in test image")
                    if hasattr(boxes, 'cls'):
                        classes = boxes.cls.int().cpu().tolist()
                        print(f"Detected classes: {classes}")
                        if hasattr(self.model, 'names'):
                            class_names = [self.model.names[int(c)] for c in classes]
                            print(f"Class names: {class_names}")
                else:
                    print("Model test: No objects detected in test image")
            else:
                print("Model test: No results from test inference")
        except Exception as e:
            print(f"Model test failed: {e}")
        """
        # Create ByteTrack configuration file if it doesn't exist
        self.tracker_config = self._create_tracker_config()
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize trajectory storage for visualization
        self.trajectories = {}
        self.max_trajectory_points = 30
        
        # Display settings
        self.show_fps = True
        self.show_trajectories = True
        self.show_labels = True
        
        # Store class indices for basketball (if available in the model)
        self.basketball_class_indices = []
        if hasattr(self.model, 'names'):
            for idx, name in self.model.names.items():
                # Look for any ball-like objects in the model's classes
                if ('ball' in name.lower() or 
                    'basketball' in name.lower() or 
                    'sports ball' in name.lower()):
                    self.basketball_class_indices.append(idx)
                    print(f"Found ball class: {idx} - {name}")
            
            if self.basketball_class_indices:
                print(f"Basketball class indices: {self.basketball_class_indices}")
            else:
                print("No basketball-related classes found in model. Will detect all objects.")
                # If no basketball classes found, we'll detect all objects
                for idx, name in self.model.names.items():
                    self.basketball_class_indices.append(idx)
        
    def _create_tracker_config(self):
        """
        Create a ByteTrack configuration file optimized for basketball tracking.
        
        Returns:
            str: Path to the configuration file
        """
        config_path = 'basketball_bytetrack.yaml'
        
        # Only create the file if it doesn't exist
        if not os.path.exists(config_path):
            config = {
                'tracker_type': 'bytetrack',
                'track_high_thresh': 0.5,    # High confidence threshold
                'track_low_thresh': 0.1,     # Low confidence threshold (key optimization)
                'new_track_thresh': 0.6,     # New track creation threshold
                'track_buffer': 40,          # Track buffer (for fast-moving basketball)
                'match_thresh': 0.8,         # First stage matching threshold
                'proximity_thresh': 0.5      # Second stage matching threshold
                # Removed problematic parameters that might not be compatible with current Ultralytics version
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
                
        return config_path
        
    def track_video(self, video_path, output_path=None):
        """
        Track basketballs in a video file.
        
        Args:
            video_path (str): Path to the input video
            output_path (str, optional): Path to save the output video
            
        Returns:
            list: Tracking results
        """
        # Start performance monitoring
        self.performance_monitor.reset()
        
        try:
            # Run tracking with ByteTrack
            results = self.model.track(
                source=video_path,
                tracker=self.tracker_config,
                conf=0.25,          # Detection confidence (lower to capture more candidates)
                iou=0.5,            # NMS IoU threshold
                show=True,
                save=True if output_path else False,
                save_txt=True,      # Save tracking results
                save_conf=True,     # Save confidence scores
                verbose=False       # Reduce console output for better performance
            )
            
            # Generate performance report
            stats = self.performance_monitor.get_stats()
            print(f"Tracking Performance: {stats}")
            
            return results
        except Exception as e:
            print(f"Error during video tracking: {e}")
            return None
    
    def track_realtime(self, source=0, display_scale=1.0):
        """
        Track basketballs in real-time using a webcam or video stream.
        
        Args:
            source (int or str): Camera index or video stream URL
            display_scale (float): Scale factor for display window
            
        Returns:
            None
        """
        # Clear trajectories
        self.trajectories = {}
        
        # Create video capture object
        cap = cv2.VideoCapture("data/video/travel.mov")
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate display dimensions
        display_width = int(frame_width * display_scale)
        display_height = int(frame_height * display_scale)
        
        print(f"Starting real-time tracking with resolution: {frame_width}x{frame_height}")
        print("Press 'q' to quit, 't' to toggle trajectories, 'f' to toggle FPS display, 'l' to toggle labels, 'd' to switch to detection mode")
        print("Press '+' to increase confidence threshold, '-' to decrease it")
        
        # Flag to switch between tracking and detection modes
        use_detection_mode = True  # Start with detection mode by default
        conf_threshold = 0.1  # Start with a very low confidence threshold
        
        # Debug counter for frames
        frame_count = 0
        detection_count = 0
        
        while True:
            # Start timing for FPS calculation
            self.performance_monitor.start_frame()
            
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            # Create a copy of the frame for processing
            processed_frame = frame.copy()
            frame_count += 1
            
            # Print debug info every 30 frames
            debug_frame = (frame_count % 30 == 0)
            
            try:
                if use_detection_mode:
                    # Run detection on the frame (no tracking)
                    results = self.model(frame, conf=conf_threshold, show=False, verbose=False)
                    if results and len(results) > 0:
                        if debug_frame and hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'cls'):
                            classes = results[0].boxes.cls.int().cpu().tolist() if hasattr(results[0].boxes, 'cls') else []
                            confs = results[0].boxes.conf.float().cpu().tolist() if hasattr(results[0].boxes, 'conf') else []
                            print(f"Detection: Found {len(classes)} objects, classes: {classes}, confidences: {[round(c, 2) for c in confs]}")
                            if classes:
                                detection_count += 1
                        
                        processed_frame = self._process_and_visualize_detections(processed_frame, results[0])
                else:
                    # Try tracking first
                    try:
                        results = self.model.track(
                            source=frame,
                            tracker=self.tracker_config,
                            conf=conf_threshold,  # Use variable confidence threshold
                            show=False,
                            verbose=False
                        )
                        
                        if results and len(results) > 0:
                            if debug_frame and hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'cls'):
                                classes = results[0].boxes.cls.int().cpu().tolist() if hasattr(results[0].boxes, 'cls') else []
                                track_ids = results[0].boxes.id.int().cpu().tolist() if hasattr(results[0].boxes, 'id') else []
                                print(f"Tracking: Found {len(classes)} objects, classes: {classes}, track IDs: {track_ids}")
                                if classes:
                                    detection_count += 1
                            
                            processed_frame = self._process_and_visualize_results(processed_frame, results[0])
                        else:
                            # If tracking returns no results, fall back to detection
                            results = self.model(frame, conf=conf_threshold, show=False, verbose=False)
                            if results and len(results) > 0:
                                processed_frame = self._process_and_visualize_detections(processed_frame, results[0])
                    except Exception as e:
                        # If tracking fails, fall back to detection
                        print(f"Warning: Tracking error - {e}. Falling back to detection.")
                        results = self.model(frame, conf=conf_threshold, show=False, verbose=False)
                        if results and len(results) > 0:
                            processed_frame = self._process_and_visualize_detections(processed_frame, results[0])
            except Exception as e:
                print(f"Warning: Processing error - {e}")
            
            # Add mode indicator and confidence threshold
            mode_text = "Detection Mode" if use_detection_mode else "Tracking Mode"
            cv2.putText(
                processed_frame,
                f"{mode_text} (conf: {conf_threshold:.2f})",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            # Add detection rate
            if frame_count > 0:
                cv2.putText(
                    processed_frame,
                    f"Detection rate: {detection_count}/{frame_count} ({100*detection_count/frame_count:.1f}%)",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
            
            # Resize for display if needed
            if display_scale != 1.0:
                processed_frame = cv2.resize(processed_frame, (display_width, display_height))
            
            # Calculate and display FPS
            fps = self.performance_monitor.end_frame()
            if self.show_fps:
                cv2.putText(
                    processed_frame, 
                    f"FPS: {fps:.1f}", 
                    (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
            
            # Display the frame
            cv2.imshow("Basketball Tracking", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.show_trajectories = not self.show_trajectories
                print(f"Trajectories: {'On' if self.show_trajectories else 'Off'}")
            elif key == ord('f'):
                self.show_fps = not self.show_fps
                print(f"FPS Display: {'On' if self.show_fps else 'Off'}")
            elif key == ord('l'):
                self.show_labels = not self.show_labels
                print(f"Labels: {'On' if self.show_labels else 'Off'}")
            elif key == ord('d'):
                use_detection_mode = not use_detection_mode
                print(f"Switched to {'Detection' if use_detection_mode else 'Tracking'} mode")
            elif key == ord('+') or key == ord('='):
                conf_threshold = min(0.9, conf_threshold + 0.05)
                print(f"Increased confidence threshold to {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"Decreased confidence threshold to {conf_threshold:.2f}")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance statistics
        stats = self.performance_monitor.get_stats()
        print(f"Performance Summary:")
        print(f"  Average FPS: {stats['avg_fps']:.2f}")
        print(f"  Min FPS: {stats['min_fps']:.2f}")
        print(f"  Max FPS: {stats['max_fps']:.2f}")
    
    def _process_and_visualize_results(self, frame, result):
        """
        Process tracking results and visualize them on the frame.
        
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
            classes = result.boxes.cls.int().cpu().tolist() if hasattr(result.boxes, 'cls') else []
            
            # Process each detected object
            for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):
                # If we have class information and basketball class indices are defined,
                # only process basketballs if we have specific basketball classes
                if classes and self.basketball_class_indices and len(self.basketball_class_indices) < len(self.model.names) and classes[i] not in self.basketball_class_indices:
                    continue
                    
                # Extract box coordinates
                x_center, y_center, width, height = box
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
                    class_name = ""
                    if classes and i < len(classes) and hasattr(self.model, 'names'):
                        class_idx = classes[i]
                        if class_idx in self.model.names:
                            class_name = f"{self.model.names[class_idx]} "
                    
                    label = f"{class_name}ID:{track_id} {conf:.2f}"
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
                    
                # Draw center point
                cv2.circle(vis_frame, (int(x_center), int(y_center)), 4, color, -1)
        except Exception as e:
            print(f"Warning: Error processing tracking results - {e}")
            # Return the original frame if processing fails
        
        return vis_frame
    
    def _draw_trajectory(self, frame, trajectory, color):
        """
        Draw the trajectory of a tracked object.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            trajectory (deque): Queue of trajectory points
            color (tuple): RGB color for the trajectory
        """
        # Need at least 2 points to draw a line
        if len(trajectory) < 2:
            return
            
        # Draw lines connecting trajectory points
        for i in range(1, len(trajectory)):
            cv2.line(
                frame,
                trajectory[i-1],
                trajectory[i],
                color,
                thickness=2
            )
    
    def _process_and_visualize_detections(self, frame, result):
        """
        Process detection results (without tracking) and visualize them on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            result (ultralytics.engine.results.Results): Detection results
            
        Returns:
            numpy.ndarray: Processed frame with visualizations
        """
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        try:
            # Check if we have valid detection results
            if result is None or not hasattr(result, 'boxes') or result.boxes is None:
                return vis_frame
                
            # Extract detection information safely
            boxes = result.boxes.xywh.cpu().numpy() if hasattr(result.boxes, 'xywh') else []
            confidences = result.boxes.conf.float().cpu().tolist() if hasattr(result.boxes, 'conf') else []
            classes = result.boxes.cls.int().cpu().tolist() if hasattr(result.boxes, 'cls') else []
            
            # Process each detected object
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                # If we have class information and basketball class indices are defined,
                # only process basketballs if we have specific basketball classes
                if classes and self.basketball_class_indices and len(self.basketball_class_indices) < len(self.model.names) and classes[i] not in self.basketball_class_indices:
                    continue
                    
                # Extract box coordinates
                x_center, y_center, width, height = box
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                # Generate a color based on class
                color = (0, 255, 0)  # Default green
                if classes and i < len(classes):
                    color = self._get_color_by_id(classes[i])
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label if enabled
                if self.show_labels:
                    class_name = ""
                    if classes and i < len(classes) and hasattr(self.model, 'names'):
                        class_idx = classes[i]
                        if class_idx in self.model.names:
                            class_name = f"{self.model.names[class_idx]} "
                    
                    label = f"{class_name}{conf:.2f}"
                    cv2.putText(
                        vis_frame, 
                        label, 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2
                    )
                
                # Draw center point
                cv2.circle(vis_frame, (int(x_center), int(y_center)), 4, color, -1)
        except Exception as e:
            print(f"Warning: Error processing detection results - {e}")
            # Return the original frame if processing fails
        
        return vis_frame
        
    def _get_color_by_id(self, track_id):
        """
        Generate a consistent color based on track ID.
        
        Args:
            track_id (int): Tracking ID
            
        Returns:
            tuple: RGB color
        """
        # Generate a deterministic color based on the track ID
        np.random.seed(int(track_id * 9999))
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        
        # Ensure the color is bright enough to be visible
        if (r + g + b) < 300:
            r = min(255, r + 100)
            g = min(255, g + 100)
            b = min(255, b + 100)
            
        return (int(b), int(g), int(r))  # OpenCV uses BGR
