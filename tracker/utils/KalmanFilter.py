"""
Kalman Filter implementation for basketball tracking
Optimized for basketball motion prediction with physics-aware filtering
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    FILTERPY_AVAILABLE = True
except ImportError:
    logger.warning("filterpy not available, using simple Kalman filter implementation")
    FILTERPY_AVAILABLE = False


class BasketballKalmanFilter:
    """
    Kalman Filter specifically designed for basketball tracking
    Handles 2D position and velocity with physics-aware motion model
    """
    
    def __init__(self, dt: float = 1.0):
        """
        Initialize basketball Kalman filter
        
        Args:
            dt: Time step between measurements (default: 1.0 frame)
        """
        self.dt = dt
        self.initialized = False
        
        if FILTERPY_AVAILABLE:
            self._init_filterpy_kalman()
        else:
            self._init_simple_kalman()
    
    def _init_filterpy_kalman(self):
        """Initialize using filterpy library (preferred)"""
        # State vector: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Measurement noise covariance
        self.kf.R = np.eye(2) * 10.0  # Measurement uncertainty
        
        # Process noise covariance
        self.kf.Q = Q_discrete_white_noise(
            dim=2, dt=self.dt, var=100.0, block_size=2
        )
        
        # Initial covariance
        self.kf.P *= 1000.0
    
    def _init_simple_kalman(self):
        """Initialize simple Kalman filter implementation"""
        # State vector: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Covariance matrices
        self.P = np.eye(4) * 1000.0  # Initial covariance
        self.Q = np.eye(4) * 100.0   # Process noise
        self.R = np.eye(2) * 10.0    # Measurement noise
    
    def initialize(self, initial_position: np.ndarray):
        """
        Initialize filter with first measurement
        
        Args:
            initial_position: Initial [x, y] position
        """
        if FILTERPY_AVAILABLE:
            self.kf.x = np.array([
                initial_position[0], initial_position[1], 0.0, 0.0
            ], dtype=np.float32)
        else:
            self.state = np.array([
                initial_position[0], initial_position[1], 0.0, 0.0
            ], dtype=np.float32)
        
        self.initialized = True
        logger.debug(f"Kalman filter initialized at position: {initial_position}")
    
    def predict(self) -> Optional[np.ndarray]:
        """
        Predict next state
        
        Returns:
            Predicted state [x, y, vx, vy] or None if not initialized
        """
        if not self.initialized:
            return None
        
        try:
            if FILTERPY_AVAILABLE:
                self.kf.predict()
                return self.kf.x.copy()
            else:
                # Simple prediction
                self.state = self.F @ self.state
                self.P = self.F @ self.P @ self.F.T + self.Q
                return self.state.copy()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def update(self, measurement: np.ndarray):
        """
        Update filter with measurement
        
        Args:
            measurement: Measured [x, y] position
        """
        if not self.initialized:
            self.initialize(measurement)
            return
        
        try:
            if FILTERPY_AVAILABLE:
                self.kf.update(measurement)
            else:
                # Simple update
                y = measurement - (self.H @ self.state)  # Innovation
                S = self.H @ self.P @ self.H.T + self.R   # Innovation covariance
                K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
                
                self.state = self.state + K @ y
                self.P = (np.eye(4) - K @ self.H) @ self.P
        except Exception as e:
            logger.error(f"Update failed: {e}")
    
    def get_position(self) -> Optional[Tuple[float, float]]:
        """
        Get current estimated position
        
        Returns:
            Current (x, y) position or None if not initialized
        """
        if not self.initialized:
            return None
        
        if FILTERPY_AVAILABLE:
            return (float(self.kf.x[0]), float(self.kf.x[1]))
        else:
            return (float(self.state[0]), float(self.state[1]))
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Get current estimated velocity
        
        Returns:
            Current (vx, vy) velocity or None if not initialized
        """
        if not self.initialized:
            return None
        
        if FILTERPY_AVAILABLE:
            return (float(self.kf.x[2]), float(self.kf.x[3]))
        else:
            return (float(self.state[2]), float(self.state[3]))
    
    def get_state(self) -> Optional[np.ndarray]:
        """
        Get full state vector
        
        Returns:
            State vector [x, y, vx, vy] or None if not initialized
        """
        if not self.initialized:
            return None
        
        if FILTERPY_AVAILABLE:
            return self.kf.x.copy()
        else:
            return self.state.copy()


class PoseKalmanFilter:
    """
    Kalman Filter for pose center tracking
    Simplified version for pose estimation smoothing
    """
    
    def __init__(self, dt: float = 1.0):
        """Initialize pose Kalman filter"""
        self.dt = dt
        self.initialized = False
        
        # Simple 2D position tracking
        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.position_variance = 1000.0
        self.velocity_variance = 100.0
        self.measurement_noise = 10.0
    
    def initialize(self, initial_position: np.ndarray):
        """Initialize with first measurement"""
        self.position = initial_position.astype(np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.initialized = True
    
    def predict(self) -> Optional[np.ndarray]:
        """Predict next position"""
        if not self.initialized:
            return None
        
        # Simple prediction with constant velocity
        predicted_position = self.position + self.velocity * self.dt
        return np.array([predicted_position[0], predicted_position[1], 
                        self.velocity[0], self.velocity[1]])
    
    def update(self, measurement: np.ndarray):
        """Update with measurement"""
        if not self.initialized:
            self.initialize(measurement)
            return
        
        # Simple smoothing
        alpha = 0.7  # Smoothing factor
        new_velocity = (measurement - self.position) / self.dt
        
        self.position = alpha * measurement + (1 - alpha) * self.position
        self.velocity = alpha * new_velocity + (1 - alpha) * self.velocity
    
    def get_position(self) -> Optional[Tuple[float, float]]:
        """Get current position"""
        if not self.initialized:
            return None
        return (float(self.position[0]), float(self.position[1]))
    
    def get_state(self) -> Optional[np.ndarray]:
        """Get full state"""
        if not self.initialized:
            return None
        return np.array([self.position[0], self.position[1], 
                        self.velocity[0], self.velocity[1]])
