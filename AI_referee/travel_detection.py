#!/usr/bin/env python3
"""
Basketball Travel Violation Detector

Detects travel violations in basketball by tracking player steps and dribbles.
Outputs violation signals based on holding state, dribbling, and step counting.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TravelViolationDetector:
    """
    Basketball Travel Violation Detector
    
    Features:
    - Detects travel violations based on steps taken while holding the ball
    - Integrates with holding detection for accurate violation identification
    - Supports different basketball rules (FIBA, NBA, etc.)
    - Provides violation signals and timestamps
    """
    
    def __init__(self, rules="FIBA"):
        """
        Initialize the travel violation detector
        
        Args:
            rules: Basketball rules to follow (FIBA, NBA)
        """
        # Initialize travel detection parameters
        self.rules = rules
        self.step_threshold = 5  # Minimum movement to count as a step
        self.steps_allowed = 2 if rules == "FIBA" else 2.5  # NBA allows gather step
        
        # Initialize counters and state
        self.step_count = 0
        self.total_step_count = 0
        self.dribble_count = 0
        self.total_dribble_count = 0
        self.travel_detected = False
        self.travel_timestamp = None
        self.travel_cooldown = 3.0  # seconds
        
        # Ball holding detection state
        self.is_holding = False
        self.holding_start_time = None
        
        # Debug info
        self.debug_info = {}
    
    def reset(self):
        """
        Reset the detection state
        """
        self.step_count = 0
        self.dribble_count = 0
        self.travel_detected = False
        self.travel_timestamp = None
        self.is_holding = False
        self.holding_start_time = None
    
    def update_step_count(self, step_detected: bool):
        """
        Update step count based on external step detection
        
        Args:
            step_detected: Boolean indicating if a step was detected
        """
        if step_detected:
            self.step_count += 1
            self.total_step_count += 1
            logger.info(f"Step taken: {self.step_count}")
            
            # Check for travel violation
            self._check_travel_violation()
    
    def update_dribble_count(self, dribble_detected: bool):
        """
        Update dribble count based on external dribble detection
        
        Args:
            dribble_detected: Boolean indicating if a dribble was detected
        """
        if dribble_detected:
            self.dribble_count += 1
            self.total_dribble_count += 1
            logger.info(f"Dribble detected: {self.total_dribble_count}")
            
            # Reset holding state and step count when dribble is detected
            self.is_holding = False
            self.step_count = 0
    
    def update_holding_state(self, is_holding: bool, timestamp=None):
        """
        Update the ball holding state from external holding detector
        
        Args:
            is_holding: Boolean indicating if the ball is being held
            timestamp: Optional timestamp of the holding state change
        """
        # If holding state changes from False to True, record the start time
        if not self.is_holding and is_holding:
            self.holding_start_time = timestamp if timestamp else time.time()
            logger.info("Ball holding started")
            
            # Reset step count when player starts holding the ball
            self.step_count = 0
        
        # If holding state changes from True to False, check for travel violation
        elif self.is_holding and not is_holding:
            logger.info("Ball holding ended")
        
        self.is_holding = is_holding
        
        # Update debug info
        self.debug_info['holding'] = is_holding
        self.debug_info['step_count'] = self.step_count
        self.debug_info['dribble_count'] = self.dribble_count
    
    def _check_travel_violation(self):
        """
        Check for travel violation based on steps while holding
        """
        # Check if player has taken too many steps while holding the ball
        if self.step_count > self.steps_allowed and self.is_holding:
            if not self.travel_detected:
                logger.info(f"Travel violation detected! Steps: {self.step_count}, Allowed: {self.steps_allowed}")
                self.travel_detected = True
                self.travel_timestamp = time.time()
                
                # Reset step count after violation
                self.step_count = 0
    
    def get_violation_status(self) -> Dict[str, Any]:
        """
        Get the current violation status
        
        Returns:
            Dict with violation status information
        """
        return {
            'violation_type': 'travel' if self.travel_detected else None,
            'timestamp': self.travel_timestamp,
            'step_count': self.step_count,
            'total_step_count': self.total_step_count,
            'dribble_count': self.dribble_count,
            'total_dribble_count': self.total_dribble_count,
            'is_holding': self.is_holding,
            'debug_info': self.debug_info
        }


if __name__ == "__main__":
    # Example usage
    detector = TravelViolationDetector()
    
    # Simulate holding the ball
    detector.update_holding_state(True)
    
    # Simulate taking steps
    detector.update_step_count(True)  # Step 1
    detector.update_step_count(True)  # Step 2
    detector.update_step_count(True)  # Step 3 - Should trigger violation
    
    # Check violation status
    status = detector.get_violation_status()
    print(f"Violation detected: {status['violation_type']}")
    print(f"Step count: {status['step_count']}")
    print(f"Is holding: {status['is_holding']}")
