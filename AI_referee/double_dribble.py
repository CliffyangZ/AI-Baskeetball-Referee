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
            Dict with violation status information
        """
        # Clear violation after cooldown
        if (self.violation_detected and self.violation_timestamp and 
            time.time() - self.violation_timestamp > self.violation_cooldown):
            self.violation_detected = False
        
        return {
            'violation_type': 'double_dribble' if self.violation_detected else None,
            'timestamp': self.violation_timestamp,
            'dribble_count': self.dribble_count,
            'dribble_sequence_active': self.dribble_sequence_active,
            'dribble_ended': self.dribble_ended,
            'is_holding': self.is_holding,
            'debug_info': self.debug_info
        }


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
