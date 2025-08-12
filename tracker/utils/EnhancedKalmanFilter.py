"""
Enhanced Kalman Filter for basketball tracking with physics-aware motion model
Handles complex basketball motion including gravity, bouncing, and occlusion
"""

import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    FILTERPY_AVAILABLE = True
except ImportError:
    logger.warning("filterpy not available, using enhanced simple implementation")
    FILTERPY_AVAILABLE = False


class EnhancedBasketballKalmanFilter:
    """
    Enhanced Kalman Filter for basketball tracking with physics-aware motion model
    Features:
    - Gravity-aware motion model for vertical movement
    - Adaptive process noise for different motion phases
    - Occlusion handling with confidence-based updates
    - Trajectory history for better prediction during occlusion
    """
    
    def __init__(self, dt: float = 1.0, gravity: float = 9.8):
        """
        Initialize enhanced basketball Kalman filter
        
        Args:
            dt: Time step between measurements (default: 1.0 frame)
            gravity: Gravity acceleration in pixels/frame^2 (adjust based on video scale)
        """
        self.dt = dt
        self.gravity = gravity * (dt ** 2)  # Convert to pixels per frame squared
        self.initialized = False
        self.occlusion_frames = 0
        self.max_occlusion_frames = 10  # Maximum frames to predict during occlusion
        
        # Motion state tracking
        self.motion_phase = "normal"  # normal, bouncing, fast_motion
        self.trajectory_history = []  # Store recent positions for pattern analysis
        self.max_history = 10
        
        if FILTERPY_AVAILABLE:
            self._init_enhanced_filterpy_kalman()
        else:
            self._init_enhanced_simple_kalman()
    
    def _init_enhanced_filterpy_kalman(self):
        """Initialize enhanced filterpy Kalman filter with gravity"""
        # State vector: [x, y, vx, vy, ax, ay] - position, velocity, acceleration
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        
        # Enhanced state transition matrix with acceleration
        self.kf.F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 0.8, 0],  # Acceleration decay (damping)
            [0, 0, 0, 0, 0, 0.8]
        ], dtype=np.float32)
        
        # Measurement matrix (observe position only)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Adaptive measurement noise
        self.base_measurement_noise = 5.0
        self.kf.R = np.eye(2) * self.base_measurement_noise
        
        # Enhanced process noise with physics awareness
        self.kf.Q = self._create_enhanced_process_noise()
        
        # Initial covariance
        self.kf.P *= 500.0
    
    def _init_enhanced_simple_kalman(self):
        """Initialize enhanced simple Kalman filter"""
        # State vector: [x, y, vx, vy, ax, ay]
        self.state = np.zeros(6, dtype=np.float32)
        
        # Enhanced state transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 0.8, 0],
            [0, 0, 0, 0, 0, 0.8]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Covariance matrices
        self.P = np.eye(6) * 500.0
        self.Q = self._create_enhanced_process_noise_simple()
        self.R = np.eye(2) * self.base_measurement_noise
    
    def _create_enhanced_process_noise(self):
        """Create physics-aware process noise matrix"""
        # Different noise levels for different state components
        pos_noise = 10.0    # Position uncertainty
        vel_noise = 50.0    # Velocity uncertainty
        acc_noise = 100.0   # Acceleration uncertainty (higher for basketball)
        
        Q = np.diag([pos_noise, pos_noise, vel_noise, vel_noise, acc_noise, acc_noise])
        return Q.astype(np.float32)
    
    def _create_enhanced_process_noise_simple(self):
        """Create enhanced process noise for simple implementation"""
        return np.diag([10.0, 10.0, 50.0, 50.0, 100.0, 100.0]).astype(np.float32)
    
    def _analyze_motion_phase(self, position: np.ndarray):
        """Analyze current motion phase based on trajectory history"""
        if len(self.trajectory_history) < 3:
            return "normal"
        
        # Calculate recent velocity changes
        recent_positions = np.array(self.trajectory_history[-3:])
        velocities = np.diff(recent_positions, axis=0)
        
        if len(velocities) >= 2:
            acceleration = np.diff(velocities, axis=0)[0]
            speed = np.linalg.norm(velocities[-1])
            
            # Detect bouncing (sudden direction change in y)
            if abs(acceleration[1]) > 50 and velocities[-1][1] * velocities[-2][1] < 0:
                return "bouncing"
            
            # Detect fast motion
            if speed > 30:
                return "fast_motion"
        
        return "normal"
    
    def _apply_gravity_correction(self):
        """Apply gravity correction to prediction"""
        if FILTERPY_AVAILABLE:
            # Add gravity to y-acceleration
            self.kf.x[5] += self.gravity  # ay += gravity
        else:
            self.state[5] += self.gravity
    
    def _adapt_noise_to_motion_phase(self):
        """Adapt noise parameters based on motion phase"""
        noise_multiplier = {
            "normal": 1.0,
            "bouncing": 3.0,    # Higher uncertainty during bouncing
            "fast_motion": 2.0  # Higher uncertainty during fast motion
        }
        
        multiplier = noise_multiplier.get(self.motion_phase, 1.0)
        
        if FILTERPY_AVAILABLE:
            self.kf.R = np.eye(2) * (self.base_measurement_noise * multiplier)
        else:
            self.R = np.eye(2) * (self.base_measurement_noise * multiplier)
    
    def initialize(self, initial_position: np.ndarray):
        """Initialize filter with first measurement"""
        if FILTERPY_AVAILABLE:
            self.kf.x = np.array([
                initial_position[0], initial_position[1], 0.0, 0.0, 0.0, self.gravity
            ], dtype=np.float32)
        else:
            self.state = np.array([
                initial_position[0], initial_position[1], 0.0, 0.0, 0.0, self.gravity
            ], dtype=np.float32)
        
        self.initialized = True
        self.trajectory_history = [initial_position.copy()]
        self.occlusion_frames = 0
        
        logger.debug(f"Enhanced Kalman filter initialized at position: {initial_position}")
    
    def predict(self) -> Optional[np.ndarray]:
        """Enhanced prediction with physics awareness"""
        if not self.initialized:
            return None
        
        try:
            # Apply gravity correction
            self._apply_gravity_correction()
            
            # Adapt noise based on motion phase
            self._adapt_noise_to_motion_phase()
            
            if FILTERPY_AVAILABLE:
                self.kf.predict()
                predicted_state = self.kf.x.copy()
            else:
                # Enhanced prediction
                self.state = self.F @ self.state
                self.P = self.F @ self.P @ self.F.T + self.Q
                predicted_state = self.state.copy()
            
            # Track occlusion frames
            self.occlusion_frames += 1
            
            return predicted_state
            
        except Exception as e:
            logger.error(f"Enhanced prediction failed: {e}")
            return None
    
    def update(self, measurement: np.ndarray, confidence: float = 1.0):
        """
        Enhanced update with confidence-based measurement
        
        Args:
            measurement: Measured [x, y] position
            confidence: Detection confidence (0.0 to 1.0)
        """
        if not self.initialized:
            self.initialize(measurement)
            return
        
        try:
            # Update trajectory history
            self.trajectory_history.append(measurement.copy())
            if len(self.trajectory_history) > self.max_history:
                self.trajectory_history.pop(0)
            
            # Analyze motion phase
            self.motion_phase = self._analyze_motion_phase(measurement)
            
            # Confidence-based update
            if confidence > 0.3:  # Only update with reasonable confidence
                # Scale measurement noise inversely with confidence
                confidence_factor = max(0.1, confidence)
                
                if FILTERPY_AVAILABLE:
                    original_R = self.kf.R.copy()
                    self.kf.R = original_R / confidence_factor
                    self.kf.update(measurement)
                    self.kf.R = original_R  # Restore original noise
                else:
                    # Enhanced update with confidence
                    original_R = self.R.copy()
                    self.R = original_R / confidence_factor
                    
                    y = measurement - (self.H @ self.state)
                    S = self.H @ self.P @ self.H.T + self.R
                    K = self.P @ self.H.T @ np.linalg.inv(S)
                    
                    self.state = self.state + K @ y
                    self.P = (np.eye(6) - K @ self.H) @ self.P
                    
                    self.R = original_R  # Restore original noise
                
                # Reset occlusion counter on successful update
                self.occlusion_frames = 0
            
        except Exception as e:
            logger.error(f"Enhanced update failed: {e}")
    
    def can_predict_during_occlusion(self) -> bool:
        """Check if we can still predict during occlusion"""
        return self.occlusion_frames <= self.max_occlusion_frames
    
    def get_position(self) -> Optional[Tuple[float, float]]:
        """Get current estimated position"""
        if not self.initialized:
            return None
        
        if FILTERPY_AVAILABLE:
            return (float(self.kf.x[0]), float(self.kf.x[1]))
        else:
            return (float(self.state[0]), float(self.state[1]))
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Get current estimated velocity"""
        if not self.initialized:
            return None
        
        if FILTERPY_AVAILABLE:
            return (float(self.kf.x[2]), float(self.kf.x[3]))
        else:
            return (float(self.state[2]), float(self.state[3]))
    
    def get_acceleration(self) -> Optional[Tuple[float, float]]:
        """Get current estimated acceleration"""
        if not self.initialized:
            return None
        
        if FILTERPY_AVAILABLE:
            return (float(self.kf.x[4]), float(self.kf.x[5]))
        else:
            return (float(self.state[4]), float(self.state[5]))
    
    def get_state(self) -> Optional[np.ndarray]:
        """Get full state vector [x, y, vx, vy, ax, ay]"""
        if not self.initialized:
            return None
        
        if FILTERPY_AVAILABLE:
            return self.kf.x.copy()
        else:
            return self.state.copy()
    
    def get_motion_info(self) -> dict:
        """Get comprehensive motion information"""
        if not self.initialized:
            return {}
        
        position = self.get_position()
        velocity = self.get_velocity()
        acceleration = self.get_acceleration()
        
        return {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "motion_phase": self.motion_phase,
            "occlusion_frames": self.occlusion_frames,
            "can_predict": self.can_predict_during_occlusion(),
            "speed": np.linalg.norm(velocity) if velocity else 0.0,
            "trajectory_length": len(self.trajectory_history)
        }
