#!/usr/bin/env python3
"""
Kalman Filter-based Trajectory Prediction for Maritime Vessel Spoofing Detection

This module implements a linear Kalman filter using a constant-velocity motion model
for predicting vessel positions from AIS telemetry. Deviations between predicted and
observed positions beyond a configurable threshold indicate potential spoofing.

The filter operates on latitude/longitude coordinates with velocity components,
handling the spherical coordinate wraparound at the antimeridian (+/-180 degrees).

Reference: Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
import logging

from ..preprocessing.data_validator import AISPoint, VesselTrack

# Configure module logger
logger = logging.getLogger(__name__)


# Physical constants
EARTH_RADIUS_KM = 6371.0
KNOTS_TO_KMH = 1.852
KM_PER_DEGREE_LAT = 111.0


@dataclass
class KalmanState:
    """
    Represents the internal state of the Kalman filter at a given timestep.
    
    The state vector is [lat, lon, d_lat/dt, d_lon/dt] where velocities
    are in degrees per hour for numerical stability.
    """
    x: np.ndarray                # State estimate [4x1]
    P: np.ndarray                # State covariance [4x4]
    timestamp: datetime          # Time of last update
    innovation: float = 0.0      # Last measurement residual (km)
    
    def __post_init__(self):
        if self.x.shape != (4,):
            raise ValueError(f"State vector must be shape (4,), got {self.x.shape}")
        if self.P.shape != (4, 4):
            raise ValueError(f"Covariance matrix must be shape (4,4), got {self.P.shape}")


@dataclass 
class KalmanConfig:
    """Configuration parameters for the Kalman filter."""
    process_noise_position: float = 1e-5      # Q diagonal for position states
    process_noise_velocity: float = 1e-5      # Q diagonal for velocity states
    measurement_noise: float = 1e-4           # R diagonal elements
    initial_covariance: float = 1.0           # Initial P diagonal
    gap_threshold_minutes: float = 7.0        # Reset filter after this gap
    deviation_threshold_km: float = 5.0       # Spoofing detection threshold


class MaritimeKalmanFilter:
    """
    Linear Kalman filter for maritime vessel trajectory prediction.
    
    Uses a constant-velocity motion model appropriate for vessels where
    acceleration/deceleration occurs on timescales longer than AIS reporting
    intervals (typically 3-5 minutes).
    
    The filter state is:
        x = [latitude, longitude, lat_velocity, lon_velocity]^T
        
    With state transition:
        x(t+dt) = F(dt) * x(t) + w
        
    Where F is the constant-velocity transition matrix and w is process noise.
    
    Attributes:
        config: KalmanConfig with filter parameters
        state: Current KalmanState or None if not initialized
        _F: State transition matrix template
        _H: Measurement matrix (observes lat/lon only)
        _Q: Process noise covariance
        _R: Measurement noise covariance
    """
    
    def __init__(self, config: Optional[KalmanConfig] = None):
        """
        Initialize the Kalman filter with the given configuration.
        
        Args:
            config: KalmanConfig instance, or None for defaults
        """
        self.config = config or KalmanConfig()
        self.state: Optional[KalmanState] = None
        
        # Build static matrices
        self._H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        self._Q = np.diag([
            self.config.process_noise_position,
            self.config.process_noise_position,
            self.config.process_noise_velocity,
            self.config.process_noise_velocity
        ])
        
        self._R = np.eye(2) * self.config.measurement_noise
        
        # F matrix template (dt-dependent entries filled at runtime)
        self._F_template = np.eye(4)
        
    def _compute_F(self, dt_hours: float) -> np.ndarray:
        """
        Compute the state transition matrix for a given time delta.
        
        The constant-velocity model gives:
            lat(t+dt) = lat(t) + d_lat/dt * dt
            lon(t+dt) = lon(t) + d_lon/dt * dt
            
        Args:
            dt_hours: Time step in hours
            
        Returns:
            4x4 state transition matrix
        """
        F = self._F_template.copy()
        F[0, 2] = dt_hours  # lat += lat_velocity * dt
        F[1, 3] = dt_hours  # lon += lon_velocity * dt
        return F
    
    @staticmethod
    def _normalize_longitude_diff(lon1: float, lon2: float) -> float:
        """
        Compute the shortest longitude difference handling antimeridian wraparound.
        
        Args:
            lon1: Reference longitude in degrees
            lon2: Target longitude in degrees
            
        Returns:
            Signed difference in degrees, in range [-180, 180]
        """
        diff = lon2 - lon1
        while diff > 180.0:
            diff -= 360.0
        while diff < -180.0:
            diff += 360.0
        return diff
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points.
        
        Uses the haversine formula for numerical stability at small distances.
        
        Args:
            lat1, lon1: First point coordinates in degrees
            lat2, lon2: Second point coordinates in degrees
            
        Returns:
            Distance in kilometers
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon1_rad = math.radians(lon1)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Handle antimeridian wraparound
        if dlon > math.pi:
            dlon -= 2 * math.pi
        elif dlon < -math.pi:
            dlon += 2 * math.pi
        
        a = (math.sin(dlat / 2.0) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2)
        c = 2.0 * math.asin(math.sqrt(a))
        
        return EARTH_RADIUS_KM * c
    
    def initialize(self, point: AISPoint, 
                   next_point: Optional[AISPoint] = None) -> None:
        """
        Initialize the filter state from an initial measurement.
        
        If a second point is provided, initial velocity is estimated from
        the two-point difference. Otherwise, velocity is initialized to zero.
        
        Args:
            point: Initial AIS measurement
            next_point: Optional second point for velocity estimation
        """
        lat0 = point.latitude
        lon0 = point.longitude
        
        if next_point is not None:
            dt_hours = (next_point.timestamp - point.timestamp).total_seconds() / 3600.0
            dt_hours = max(dt_hours, 1e-6)  # Avoid division by zero
            
            dlat = next_point.latitude - lat0
            dlon = self._normalize_longitude_diff(lon0, next_point.longitude)
            
            lat_vel = dlat / dt_hours
            lon_vel = dlon / dt_hours
        else:
            lat_vel = 0.0
            lon_vel = 0.0
        
        x = np.array([lat0, lon0, lat_vel, lon_vel])
        P = np.eye(4) * self.config.initial_covariance
        
        self.state = KalmanState(x=x, P=P, timestamp=point.timestamp)
        logger.debug(f"Filter initialized at ({lat0:.5f}, {lon0:.5f})")
    
    def predict(self, target_time: datetime) -> Tuple[float, float]:
        """
        Predict the vessel position at a future time.
        
        Applies the state transition model without incorporating any measurement.
        
        Args:
            target_time: Time to predict position for
            
        Returns:
            Tuple of (predicted_latitude, predicted_longitude)
            
        Raises:
            RuntimeError: If filter not initialized
        """
        if self.state is None:
            raise RuntimeError("Filter must be initialized before prediction")
        
        dt_seconds = (target_time - self.state.timestamp).total_seconds()
        dt_hours = dt_seconds / 3600.0
        
        if dt_hours < 0:
            logger.warning(f"Negative time delta: {dt_hours:.4f} hours")
            dt_hours = 0.0
        
        F = self._compute_F(dt_hours)
        
        # State prediction: x_pred = F * x
        x_pred = F @ self.state.x
        
        # Covariance prediction: P_pred = F * P * F^T + Q
        P_pred = F @ self.state.P @ F.T + self._Q
        
        # Update internal state
        self.state = KalmanState(
            x=x_pred,
            P=P_pred,
            timestamp=target_time,
            innovation=self.state.innovation
        )
        
        return (x_pred[0], x_pred[1])
    
    def update(self, measurement: AISPoint) -> float:
        """
        Incorporate a new measurement and compute the innovation.
        
        The innovation (measurement residual) is the geodesic distance between
        the predicted and observed positions. Large innovations indicate
        potential spoofing.
        
        Args:
            measurement: New AIS position measurement
            
        Returns:
            Innovation distance in kilometers
            
        Raises:
            RuntimeError: If filter not initialized
        """
        if self.state is None:
            raise RuntimeError("Filter must be initialized before update")
        
        # First predict to measurement time
        self.predict(measurement.timestamp)
        
        # Form measurement vector with wraparound handling
        z = np.array([measurement.latitude, measurement.longitude])
        
        # Handle longitude wraparound relative to current state
        lon_diff = self._normalize_longitude_diff(self.state.x[1], z[1])
        z[1] = self.state.x[1] + lon_diff
        
        # Innovation: y = z - H * x
        y = z - self._H @ self.state.x
        
        # Innovation covariance: S = H * P * H^T + R
        S = self._H @ self.state.P @ self._H.T + self._R
        
        # Kalman gain: K = P * H^T * S^(-1)
        K = self.state.P @ self._H.T @ np.linalg.inv(S)
        
        # State update: x = x + K * y
        x_new = self.state.x + K @ y
        
        # Covariance update: P = (I - K * H) * P
        I_KH = np.eye(4) - K @ self._H
        P_new = I_KH @ self.state.P
        
        # Compute geodesic innovation distance
        innovation_km = self.haversine_distance(
            self.state.x[0], self.state.x[1],
            measurement.latitude, measurement.longitude
        )
        
        # Update state
        self.state = KalmanState(
            x=x_new,
            P=P_new,
            timestamp=measurement.timestamp,
            innovation=innovation_km
        )
        
        return innovation_km
    
    def should_reset(self, current_time: datetime) -> bool:
        """
        Check if the filter should be reset due to a large time gap.
        
        Long gaps cause the Kalman prediction to diverge from reality,
        so we reset the filter state when gaps exceed the configured threshold.
        
        Args:
            current_time: Time of incoming measurement
            
        Returns:
            True if filter should be reset
        """
        if self.state is None:
            return True
        
        gap_minutes = (current_time - self.state.timestamp).total_seconds() / 60.0
        return gap_minutes > self.config.gap_threshold_minutes
    
    def reset(self) -> None:
        """Clear the filter state, requiring reinitialization."""
        self.state = None
        logger.debug("Filter state reset")
    
    def get_predicted_position(self) -> Optional[Tuple[float, float]]:
        """
        Get the current predicted position.
        
        Returns:
            Tuple of (latitude, longitude) or None if not initialized
        """
        if self.state is None:
            return None
        return (self.state.x[0], self.state.x[1])
    
    def get_predicted_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Get the current predicted velocity.
        
        Returns:
            Tuple of (lat_velocity, lon_velocity) in deg/hour, or None
        """
        if self.state is None:
            return None
        return (self.state.x[2], self.state.x[3])


class TrajectoryAnalyzer:
    """
    High-level trajectory analysis using Kalman filtering.
    
    Processes complete vessel tracks and identifies points with anomalous
    deviations from predicted trajectories.
    """
    
    def __init__(self, config: Optional[KalmanConfig] = None):
        self.config = config or KalmanConfig()
        self.kf = MaritimeKalmanFilter(self.config)
    
    def analyze_track(self, track: VesselTrack) -> List[Dict[str, Any]]:
        """
        Analyze a vessel track for trajectory anomalies.
        
        Args:
            track: VesselTrack containing sorted AIS points
            
        Returns:
            List of anomaly records with timestamps, positions, and deviations
        """
        if len(track.points) < 2:
            return []
        
        anomalies = []
        points = sorted(track.points, key=lambda p: p.timestamp)
        
        # Initialize filter with first two points
        self.kf.initialize(points[0], points[1])
        
        for i, point in enumerate(points[1:], start=1):
            # Check for gap requiring reset
            if self.kf.should_reset(point.timestamp):
                logger.debug(f"Resetting filter at index {i} due to time gap")
                if i + 1 < len(points):
                    self.kf.initialize(point, points[i + 1])
                else:
                    self.kf.initialize(point)
                continue
            
            # Update filter and get innovation
            innovation_km = self.kf.update(point)
            
            # Check for anomaly
            if innovation_km > self.config.deviation_threshold_km:
                pred_pos = self.kf.get_predicted_position()
                anomalies.append({
                    'index': i,
                    'timestamp': point.timestamp.isoformat(),
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'predicted_lat': pred_pos[0] if pred_pos else None,
                    'predicted_lon': pred_pos[1] if pred_pos else None,
                    'deviation_km': innovation_km,
                    'reason': 'deviation'
                })
        
        return anomalies


# Factory function for external use
def create_kalman_filter(
    deviation_threshold_km: float = 5.0,
    gap_threshold_minutes: float = 7.0,
    process_noise: float = 1e-5,
    measurement_noise: float = 1e-4
) -> MaritimeKalmanFilter:
    """
    Create a configured Kalman filter instance.
    
    Args:
        deviation_threshold_km: Threshold for flagging anomalies
        gap_threshold_minutes: Gap threshold for filter reset
        process_noise: Process noise covariance diagonal
        measurement_noise: Measurement noise covariance diagonal
        
    Returns:
        Configured MaritimeKalmanFilter instance
    """
    config = KalmanConfig(
        process_noise_position=process_noise,
        process_noise_velocity=process_noise,
        measurement_noise=measurement_noise,
        gap_threshold_minutes=gap_threshold_minutes,
        deviation_threshold_km=deviation_threshold_km
    )
    return MaritimeKalmanFilter(config)
