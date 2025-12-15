"""
Kalman filter for sperm motion prediction.

Implements constant velocity model for tracking smooth trajectories.
"""

import numpy as np
from filterpy.kalman import KalmanFilter as FilterPyKalman
from typing import Tuple


class KalmanFilter:
    """
    Kalman filter for 2D position and velocity tracking.
    
    State vector: [x, y, vx, vy]
    Measurement vector: [x, y]
    
    Assumes constant velocity model with process and measurement noise.
    """
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        dt: float = 1.0,
        process_noise: float = 0.1,
        measurement_noise: float = 0.1
    ):
        """
        Initialize Kalman filter.
        
        Args:
            initial_position: Initial (x, y) position.
            dt: Time step between measurements.
            process_noise: Process noise standard deviation.
            measurement_noise: Measurement noise standard deviation.
        """
        self.dt = dt
        
        # Create Kalman filter
        # State: [x, y, vx, vy]
        # Measurement: [x, y]
        self.kf = FilterPyKalman(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = process_noise ** 2
        self.kf.Q = np.array([
            [q * dt**4 / 4, 0, q * dt**3 / 2, 0],
            [0, q * dt**4 / 4, 0, q * dt**3 / 2],
            [q * dt**3 / 2, 0, q * dt**2, 0],
            [0, q * dt**3 / 2, 0, q * dt**2]
        ])
        
        # Measurement noise covariance
        r = measurement_noise ** 2
        self.kf.R = np.array([
            [r, 0],
            [0, r]
        ])
        
        # Initial state
        x0, y0 = initial_position
        self.kf.x = np.array([x0, y0, 0, 0])
        
        # Initial covariance (high uncertainty in velocity)
        self.kf.P = np.eye(4) * 10
        self.kf.P[2:, 2:] = 100  # High uncertainty in initial velocity
    
    def predict(self) -> Tuple[float, float]:
        """
        Predict next state (position).
        
        Returns:
            Predicted (x, y) position.
        """
        self.kf.predict()
        return self.kf.x[0], self.kf.x[1]
    
    def update(self, measurement: Tuple[float, float]):
        """
        Update filter with new measurement.
        
        Args:
            measurement: Observed (x, y) position.
        """
        z = np.array([measurement[0], measurement[1]])
        self.kf.update(z)
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """
        Get current state estimate.
        
        Returns:
            Tuple of (x, y, vx, vy).
        """
        return tuple(self.kf.x)
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current position estimate.
        
        Returns:
            Tuple of (x, y).
        """
        return self.kf.x[0], self.kf.x[1]
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate.
        
        Returns:
            Tuple of (vx, vy).
        """
        return self.kf.x[2], self.kf.x[3]
    
    def get_covariance(self) -> np.ndarray:
        """
        Get state covariance matrix.
        
        Returns:
            4x4 covariance matrix.
        """
        return self.kf.P.copy()


class AdaptiveKalmanFilter(KalmanFilter):
    """
    Kalman filter with adaptive noise parameters.
    
    Adjusts process noise based on motion characteristics.
    Useful for sperm that exhibit variable swimming patterns.
    """
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        dt: float = 1.0,
        min_process_noise: float = 0.05,
        max_process_noise: float = 0.5,
        measurement_noise: float = 0.1
    ):
        """
        Initialize adaptive Kalman filter.
        
        Args:
            initial_position: Initial (x, y) position.
            dt: Time step.
            min_process_noise: Minimum process noise level.
            max_process_noise: Maximum process noise level.
            measurement_noise: Measurement noise level.
        """
        super().__init__(initial_position, dt, min_process_noise, measurement_noise)
        
        self.min_process_noise = min_process_noise
        self.max_process_noise = max_process_noise
        
        # History for adaptation
        self.innovation_history = []
        self.history_length = 10
    
    def update(self, measurement: Tuple[float, float]):
        """
        Update with adaptive process noise.
        
        Args:
            measurement: Observed (x, y) position.
        """
        # Compute innovation (measurement - prediction)
        predicted = self.get_position()
        innovation = np.sqrt(
            (measurement[0] - predicted[0])**2 +
            (measurement[1] - predicted[1])**2
        )
        
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.history_length:
            self.innovation_history.pop(0)
        
        # Adapt process noise based on innovation magnitude
        if len(self.innovation_history) >= 3:
            avg_innovation = np.mean(self.innovation_history)
            
            # High innovation → increase process noise (unpredictable motion)
            # Low innovation → decrease process noise (smooth motion)
            if avg_innovation > 5:  # Threshold in pixels
                process_noise = self.max_process_noise
            else:
                process_noise = self.min_process_noise
            
            # Update Q matrix
            dt = self.dt
            q = process_noise ** 2
            self.kf.Q = np.array([
                [q * dt**4 / 4, 0, q * dt**3 / 2, 0],
                [0, q * dt**4 / 4, 0, q * dt**3 / 2],
                [q * dt**3 / 2, 0, q * dt**2, 0],
                [0, q * dt**3 / 2, 0, q * dt**2]
            ])
        
        # Standard update
        super().update(measurement)
