import numpy as np

class BasketballKalmanFilter:
    """
    Kalman filter optimized for basketball motion characteristics.
    """
    def __init__(self):
        """
        Initialize the Kalman filter for basketball tracking.
        """
        try:
            from filterpy.kalman import KalmanFilter
            self.kf = KalmanFilter(dim_x=4, dim_z=2)
            self.initialized = False
        except ImportError:
            print("Warning: filterpy not installed. Kalman filtering disabled.")
            self.initialized = False
            self.kf = None
    
    def initialize(self, initial_state):
        """
        Initialize the Kalman filter with an initial state.
        
        Args:
            initial_state (numpy.ndarray): Initial state [x, y, vx, vy]
        """
        if self.kf is None:
            return
            
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise (adjusted for basketball tracking)
        self.kf.R = np.array([
            [10, 0],
            [0, 10]
        ])
        
        # Process noise (adjusted for basketball physics)
        self.kf.Q = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10]
        ])
        
        # Initial state
        self.kf.x = np.array([
            [initial_state[0]],  # x position
            [initial_state[1]],  # y position
            [initial_state[2]],  # x velocity
            [initial_state[3]]   # y velocity
        ])
        
        # Initial state covariance
        self.kf.P = np.array([
            [100, 0, 0, 0],
            [0, 100, 0, 0],
            [0, 0, 100, 0],
            [0, 0, 0, 100]
        ])
        
        self.initialized = True
    
    def predict(self):
        """
        Predict the next state.
        
        Returns:
            numpy.ndarray: Predicted state [x, y, vx, vy]
        """
        if not self.initialized or self.kf is None:
            return None
            
        self.kf.predict()
        return self.kf.x.flatten()
    
    def update(self, measurement):
        """
        Update the filter with a new measurement.
        
        Args:
            measurement (numpy.ndarray): New measurement [x, y]
            
        Returns:
            numpy.ndarray: Updated state [x, y, vx, vy]
        """
        if not self.initialized or self.kf is None:
            return measurement
            
        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x.flatten()
