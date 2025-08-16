#!/usr/bin/env python3
"""
Basketball Double Dribble Violation Detector

Detects double dribble violations in basketball by tracking dribbling and holding states.
Outputs violation signals based on holding state and dribble sequences.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DoubleDribbleDetector:
    """
    Basketball Double Dribble Violation Detector
    
    Features:
    - Detects double dribble violations based on dribbling and holding patterns
    - Integrates with holding detection for accurate violation identification
    - Provides violation signals and timestamps
    """
    
    def __init__(self):
        """
        Initialize the double dribble detector
        """
        # State tracking
        self.is_holding = False
        self.was_holding = False
        self.dribble_sequence_active = False
        self.dribble_ended = False
        self.last_dribble_time = None
        self.holding_start_time = None
        
        # Violation detection
        self.violation_detected = False
        self.violation_timestamp = None
        self.violation_cooldown = 3.0  # seconds
        
        # Dribble history
        self.dribble_count = 0
        self.dribble_history: List[Dict[str, Any]] = []
        
        # Basketball tracking
        self.basketball_positions = {}
        
        # Debug info
        self.debug_info = {}
    
    def reset(self):
        """
        Reset the detection state
        """
        self.is_holding = False
        self.was_holding = False
        self.dribble_sequence_active = False
        self.dribble_ended = False
        self.last_dribble_time = None
        self.holding_start_time = None
        self.violation_detected = False
        self.violation_timestamp = None
        self.dribble_count = 0
        self.dribble_history = []
    
    def update_dribble_state(self, dribble_detected: bool, timestamp=None):
        """
        Update dribble state based on external dribble detection
        
        Args:
            dribble_detected: Boolean indicating if a dribble was detected
            timestamp: Optional timestamp of the dribble detection
        """
        current_time = timestamp if timestamp else time.time()
        
        if dribble_detected:
            # Record dribble event
            self.dribble_count += 1
            self.last_dribble_time = current_time
            
            # Start dribble sequence if not already active
            if not self.dribble_sequence_active:
                self.dribble_sequence_active = True
                logger.info("Dribble sequence started")
            
            # Add to dribble history
            self.dribble_history.append({
                'timestamp': current_time,
                'holding_before': self.was_holding
            })
            
            # Check for double dribble violation
            # If we previously had a dribble sequence, then held the ball (dribble_ended),
            # and now we're dribbling again, that's a double dribble
            if self.dribble_ended and self.was_holding:
                logger.info("Double dribble condition detected: dribbling after holding")
                self.violation_detected = True
                self.violation_timestamp = current_time
            
            logger.info(f"Dribble detected: {self.dribble_count}")
    
    def update_holding_state(self, is_holding: bool, timestamp=None):
        """
        Update the ball holding state from external holding detector
        
        Args:
            is_holding: Boolean indicating if the ball is being held
            timestamp: Optional timestamp of the holding state change
        """
        current_time = timestamp if timestamp else time.time()
        
        # Save previous holding state
        self.was_holding = self.is_holding
        
        # If holding state changes from False to True, record the start time
        if not self.is_holding and is_holding:
            self.holding_start_time = current_time
            logger.info("Ball holding started")
            
            # If a dribble sequence was active and now the ball is held again,
            # mark the dribble sequence as ended
            if self.dribble_sequence_active:
                self.dribble_ended = True
                logger.info("Dribble sequence ended")
        
        # Update current holding state
        self.is_holding = is_holding
        
        # Check for double dribble violation
        self._check_double_dribble_violation()
        
        # Update debug info
        self.debug_info['holding'] = is_holding
        self.debug_info['dribble_sequence_active'] = self.dribble_sequence_active
        self.debug_info['dribble_ended'] = self.dribble_ended
    
    def _check_double_dribble_violation(self):
        """
        Check for double dribble violation based on dribbling and holding patterns
        """
        # Double dribble occurs when:
        # 1. Player dribbled (dribble_sequence_active)
        # 2. Then held the ball (dribble_ended)
        # 3. Then dribbled again (new dribble after dribble_ended)
        
        if (self.dribble_sequence_active and 
            self.dribble_ended and 
            not self.is_holding and 
            len(self.dribble_history) >= 2):
            
            # Check if the most recent dribble happened after the dribble sequence ended
            latest_dribble = self.dribble_history[-1]
            if latest_dribble['holding_before']:
                if not self.violation_detected:
                    logger.info("Double dribble violation detected!")
                    self.violation_detected = True
                    self.violation_timestamp = time.time()
    
    def get_violation_status(self) -> Dict[str, Any]:
        """
        Get the current violation status
        
        Returns:
            Dict with violation information
        """
        return {
            'violation_detected': self.violation_detected,
            'violation_timestamp': self.violation_timestamp,
            'dribble_count': self.dribble_count,
            'dribble_sequence_active': self.dribble_sequence_active,
            'dribble_ended': self.dribble_ended,
            'is_holding': self.is_holding,
            'debug_info': self.debug_info
        }
        
    def process_frame(self, frame):
        """
        Process a video frame to detect double dribble violations
        
        Args:
            frame: Video frame to process
            
        Returns:
            Tuple of (result_dict, annotated_frame)
        """
        # For the double dribble detector, we don't process frames directly
        # Instead, we rely on external dribble and holding detection
        # This method is implemented for compatibility with the referee system
        
        # Create a copy of the frame for annotations
        annotated_frame = frame.copy()
        
        # Get current violation status
        result = self.get_violation_status()
        
        # Add visualization to the frame
        self._annotate_frame(annotated_frame)
        
        return result, annotated_frame
        
    def _annotate_frame(self, frame):
        """
        Add visualization overlays to the frame
        
        Args:
            frame: Frame to annotate
        """
        import cv2
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Add violation status overlay
        status_text = []
        
        # Violation status
        if self.violation_detected:
            status_text.append("DOUBLE DRIBBLE VIOLATION DETECTED!")
            # Add red overlay for violations
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Current state
        status_text.append(f"Holding: {'Yes' if self.is_holding else 'No'}")
        status_text.append(f"Dribble Count: {self.dribble_count}")
        status_text.append(f"Dribble Sequence: {'Active' if self.dribble_sequence_active else 'Inactive'}")
        status_text.append(f"Dribble Ended: {'Yes' if self.dribble_ended else 'No'}")
        
        # Draw status text
        for i, text in enumerate(status_text):
            y_pos = 30 + (i * 30)
            # Add black outline for better visibility
            cv2.putText(frame, text, (21, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
    def process_frame_with_tracks(self, frame, basketball_tracks, pose_tracks=None):
        """
        Process a frame with pre-detected basketball and pose tracks.
        
        Args:
            frame: Input video frame
            basketball_tracks: Dictionary or list of basketball tracks from the tracker
            pose_tracks: Optional dictionary or list of pose tracks
            
        Returns:
            dict: Dictionary containing double dribble detection results
            frame: Annotated frame
        """
        try:
            # Create a copy of the frame for annotation
            annotated_frame = frame.copy() if frame is not None else None
            
            # Process basketball tracks
            if basketball_tracks:
                # Handle different track formats
                if isinstance(basketball_tracks, dict):
                    # Dictionary format with track_id as keys
                    for track_id, track_data in basketball_tracks.items():
                        # Extract position
                        center = track_data.get('center', None)
                        
                        if center:
                            # Update basketball position
                            self.basketball_positions[track_id] = center
                elif isinstance(basketball_tracks, list):
                    # List format with track objects
                    for i, track_data in enumerate(basketball_tracks):
                        # Extract track ID and center
                        track_id = track_data.get('track_id', i) if isinstance(track_data, dict) else i
                        
                        # Handle different track data formats
                        if isinstance(track_data, dict):
                            center = track_data.get('center', None)
                            
                            if center:
                                # Update basketball position
                                self.basketball_positions[track_id] = center
                        elif isinstance(track_data, tuple) and len(track_data) >= 2:
                            # Assume tuple format (center_x, center_y, ...)
                            center = (track_data[0], track_data[1])
                            self.basketball_positions[track_id] = center
            
            # Process pose tracks if available
            if pose_tracks:
                # Similar handling for different pose track formats
                if isinstance(pose_tracks, dict):
                    for person_id, pose_data in pose_tracks.items():
                        # Process pose data
                        pass
                elif isinstance(pose_tracks, list):
                    for i, pose_data in enumerate(pose_tracks):
                        # Process pose data
                        pass
            
            # Get current violation status
            result = self.get_violation_status()
            
            # Add visualization to the frame
            if annotated_frame is not None:
                self._annotate_frame(annotated_frame)
            
            return result, annotated_frame
            
        except Exception as e:
            logger.error(f"Error in process_frame_with_tracks: {e}")
            # Return empty result and original frame
            return self.get_violation_status(), frame


if __name__ == "__main__":
    # Example usage
    detector = DoubleDribbleDetector()
    
    # Simulate first dribble sequence
    detector.update_dribble_state(True)  # First dribble
    detector.update_dribble_state(True)  # Second dribble
    
    # Simulate holding the ball (ending first dribble sequence)
    detector.update_holding_state(True)
    
    # Simulate second dribble sequence (should trigger violation)
    detector.update_holding_state(False)
    detector.update_dribble_state(True)  # This should trigger violation
    
    # Check violation status
    status = detector.get_violation_status()
    print(f"Violation detected: {status['violation_type']}")
    print(f"Dribble count: {status['dribble_count']}")
    print(f"Is holding: {status['is_holding']}")
