"""
referee_integration.py
---------------------
Integration module for AI Basketball Referee with real-time statistics.
Connects the referee.py backend with the frontend interface.
"""

import sys
import os
import threading
import time
import queue
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

# Add AI_referee to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from AI_referee.referee import BasketballReferee, DeviceType

logger = logging.getLogger(__name__)

class RefereeIntegration:
    """Integration class for real-time basketball referee functionality"""
    
    def __init__(self):
        self.referee = None
        self.running = False
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=2)  # Much smaller queue
        self.result_queue = queue.Queue(maxsize=10)
        self.current_stats = {
            'shot_makes': 0,
            'shot_attempts': 0,
            'shooting_percentage': 0.0,
            'dribble_count': 0,
            'step_count': {},
            'total_steps': 0,
            'violations': [],
            'fps': 0.0,
            'processing_time': 0.0
        }
        self.annotated_frame = None
        self.lock = threading.Lock()
        self.last_process_time = 0
       
    def initialize_referee(self, 
                          basketball_model: str = None,
                          pose_model: str = None,
                          device: DeviceType = DeviceType.CPU,
                          rules: str = "FIBA") -> bool:
        """Initialize the basketball referee system"""
        try:
            # Set default model paths if not provided
            if basketball_model is None:
                basketball_model = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'ov_models', 'basketballModel_openvino_model', 'basketballModel.xml')
            if pose_model is None:
                pose_model = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'ov_models', 'yolov8s-pose_openvino_model', 'yolov8s-pose.xml')
            
            # Check if models exist, if not, run in fallback mode with mock detection
            models_available = True
            if not os.path.exists(basketball_model):
                logger.warning(f"Basketball model not found: {basketball_model}")
                models_available = False
            if not os.path.exists(pose_model):
                logger.warning(f"Pose model not found: {pose_model}")
                models_available = False
            
            if not models_available:
                logger.info("Running in fallback mode with mock detection for demo")
                # Create a mock referee for demo purposes
                self.referee = None
                self._setup_mock_detection()
                return True
                
            self.referee = BasketballReferee(
                basketball_model_path=basketball_model,
                pose_model_path=pose_model,
                video_path=0,  # Use camera 0 as default
                device=device,
                rules=rules
            )
            logger.info("Basketball referee initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize referee: {e}")
            # Fallback to mock mode on any error
            self.referee = None
            self._setup_mock_detection()
            return True
    
    def start_processing(self):
        """Start the background processing thread"""
        if self.running:
            return False
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Referee processing started")
        return True
    
    def stop_processing(self):
        """Stop the background processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        logger.info("Referee processing stopped")
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """Add a frame to the processing queue"""
        if not self.running:
            return False
            
        try:
            # Non-blocking put - drop frame if queue is full
            self.frame_queue.put_nowait(frame.copy())
            return True
        except queue.Full:
            # Drop oldest frame and add new one
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame.copy())
                return True
            except queue.Empty:
                return False
    
    def process_frame_async(self, frame: np.ndarray) -> bool:
        """Add a frame to the processing queue with aggressive frame dropping for better FPS"""
        if not self.running:
            return False
        
        # Rate limiting - don't process more than 10 FPS
        current_time = time.time()
        if current_time - self.last_process_time < 0.1:  # 100ms minimum interval
            return False
        
        try:
            # Clear entire queue to ensure only latest frame is processed
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Add new frame
            self.frame_queue.put_nowait(frame.copy())
            self.last_process_time = current_time
            return True
        except queue.Full:
            return False
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            return self.current_stats.copy()
    
    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame"""
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None
    
    def _processing_loop(self):
        """Main processing loop running in background thread"""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=0.2)  # Longer timeout
                
                # Process frame with referee (or fallback mode)
                process_start = time.time()
                if self.referee is not None:
                    annotated_frame = self.referee.process_frame(frame)
                    stats = self.referee.get_statistics()
                else:
                    # Fallback mode - mock detection with visual feedback
                    annotated_frame = self._mock_detection(frame)
                    stats = self._get_mock_stats()
                
                process_time = time.time() - process_start
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Update current stats
                with self.lock:
                    self.current_stats.update({
                        'shot_makes': stats.get('shot_makes', 0),
                        'shot_attempts': stats.get('shot_attempts', 0),
                        'shooting_percentage': stats.get('shooting_percentage', 0.0),
                        'dribble_count': stats.get('dribble_count', 0),
                        'step_count': stats.get('step_count', {}),
                        'total_steps': sum(stats.get('step_count', {}).values()),
                        'violations': stats.get('violations', []),
                        'fps': fps,
                        'processing_time': process_time * 1000  # Convert to ms
                    })
                    
                    # Store annotated frame
                    if annotated_frame is not None:
                        self.annotated_frame = annotated_frame
                
                # Add result to queue for frontend
                result = {
                    'timestamp': time.time(),
                    'stats': self.current_stats.copy(),
                    'frame_processed': True
                }
                
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Remove oldest result
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                continue
    
    def get_detection_results(self) -> Optional[Dict[str, Any]]:
        """Get latest detection results"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _setup_mock_detection(self):
        """Setup mock detection for demo mode"""
        self.mock_frame_count = 0
        self.mock_basketball_pos = [320, 240]  # Center of 640x480 frame
        self.mock_velocity = [2, 1]
        logger.info("Mock detection setup complete")
    
    def _mock_detection(self, frame):
        """Create mock detection visualization"""
        self.mock_frame_count += 1
        annotated_frame = frame.copy()
        
        # Draw mock basketball detection
        h, w = frame.shape[:2]
        
        # Update mock basketball position
        self.mock_basketball_pos[0] += self.mock_velocity[0]
        self.mock_basketball_pos[1] += self.mock_velocity[1]
        
        # Bounce off walls
        if self.mock_basketball_pos[0] <= 30 or self.mock_basketball_pos[0] >= w - 30:
            self.mock_velocity[0] *= -1
        if self.mock_basketball_pos[1] <= 30 or self.mock_basketball_pos[1] >= h - 30:
            self.mock_velocity[1] *= -1
        
        # Keep within bounds
        self.mock_basketball_pos[0] = max(30, min(w - 30, self.mock_basketball_pos[0]))
        self.mock_basketball_pos[1] = max(30, min(h - 30, self.mock_basketball_pos[1]))
        
        # Draw basketball
        center = (int(self.mock_basketball_pos[0]), int(self.mock_basketball_pos[1]))
        cv2.circle(annotated_frame, center, 20, (0, 165, 255), 3)  # Orange circle
        cv2.putText(annotated_frame, "Basketball (Mock)", 
                   (center[0] - 50, center[1] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Draw demo info
        cv2.putText(annotated_frame, "DEMO MODE - No AI Models Loaded", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, "Mock Detection Active", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_frame
    
    def _get_mock_stats(self):
        """Generate mock statistics for demo"""
        # Simulate changing stats over time
        frame_based_shots = (self.mock_frame_count // 300) % 10  # Change every 10 seconds at 30fps
        frame_based_makes = (frame_based_shots * 6) // 10  # 60% shooting
        
        return {
            'shot_makes': frame_based_makes,
            'shot_attempts': frame_based_shots,
            'shooting_percentage': (frame_based_makes / max(1, frame_based_shots)) * 100,
            'dribble_count': (self.mock_frame_count // 60) % 20,  # Change every 2 seconds
            'step_count': {0: (self.mock_frame_count // 30) % 50},  # Mock player 0 steps
            'violations': []
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_processing()
