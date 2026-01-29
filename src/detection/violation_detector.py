#!/usr/bin/env python3
"""
Violation Detection Module for Maritime GPS Spoofing Analysis

This module implements the physical and environmental constraint checks
used to identify implausible vessel behavior indicative of GPS spoofing:

1. Speed violations: Instantaneous speeds exceeding 60 knots
2. Deviation violations: Kalman residuals exceeding 5km
3. Land violations: Positions falling outside water bodies

The module integrates with rasterio for Global Surface Water (GSW)
raster lookups and with the Kalman filter for trajectory prediction.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Set
from datetime import datetime, timedelta
from enum import Enum, auto

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    rasterio = None

from .kalman_filter import MaritimeKalmanFilter, KalmanConfig
from ..preprocessing.data_validator import AISPoint

logger = logging.getLogger(__name__)


# Physical constants
EARTH_RADIUS_KM = 6371.0
MAX_SHIP_SPEED_KNOTS = 60.0     # Upper bound for plausible vessel speed
SMALL_JUMP_THRESHOLD_KM = 5.0  # Suppress violations for small displacements
COASTAL_BUFFER_KM = 1.0        # Buffer zone for water detection
GSW_WATER_THRESHOLD = 50       # Minimum water occurrence to count as water


class ViolationType(Enum):
    """Types of violations detected by the system."""
    SPEED = auto()      # Speed exceeds physical limits
    DEVIATION = auto()  # Large deviation from predicted trajectory
    LAND = auto()       # Position falls on land


@dataclass
class ViolationRecord:
    """
    Record of a detected violation at a specific point.
    
    Attributes:
        index: Index in the route array
        timestamp: ISO timestamp string
        latitude: Observed latitude
        longitude: Observed longitude
        violation_type: Type of violation detected
        value: Numeric value (speed in knots, deviation in km, etc.)
        threshold: Threshold that was exceeded
    """
    index: int
    timestamp: str
    latitude: float
    longitude: float
    violation_type: ViolationType
    value: float
    threshold: float
    
    @property
    def reason(self) -> str:
        """String representation of violation type for reporting."""
        return self.violation_type.name.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'reason': self.reason,
            'value': round(self.value, 3),
            'threshold': self.threshold
        }


class WaterMaskChecker:
    """
    Checker for land/water classification using Global Surface Water data.
    
    Uses rasterio to query the GSW occurrence raster. Positions with
    water occurrence below the threshold are classified as land.
    
    The checker maintains an open raster handle for efficient repeated queries.
    """
    
    def __init__(self, raster_path: str, 
                 threshold: int = GSW_WATER_THRESHOLD,
                 buffer_km: float = COASTAL_BUFFER_KM):
        """
        Initialize the water mask checker.
        
        Args:
            raster_path: Path to the GSW occurrence VRT/GeoTIFF
            threshold: Minimum water occurrence percentage to count as water
            buffer_km: Buffer distance for coastal checking
            
        Raises:
            ImportError: If rasterio is not available
            FileNotFoundError: If raster file cannot be opened
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for water mask checking")
        
        self.raster_path = raster_path
        self.threshold = threshold
        self.buffer_km = buffer_km
        
        self._raster: Optional[rasterio.DatasetReader] = None
        self._resolution_deg: float = 0.0003  # ~30m at equator
    
    def open(self) -> None:
        """Open the raster dataset."""
        if self._raster is not None:
            return
        try:
            self._raster = rasterio.open(self.raster_path)
            # Estimate resolution from transform
            self._resolution_deg = abs(self._raster.transform.a)
            logger.info(f"Opened GSW raster: {self.raster_path}")
        except Exception as e:
            logger.error(f"Failed to open GSW raster: {e}")
            raise
    
    def close(self) -> None:
        """Close the raster dataset."""
        if self._raster is not None:
            self._raster.close()
            self._raster = None
    
    def is_on_water(self, lat: float, lon: float) -> bool:
        """
        Check if a coordinate is on water.
        
        Uses the GSW occurrence raster to determine water presence.
        Includes a buffer zone check near coastlines to avoid false
        positives from GPS jitter.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            True if position is on water, False if on land
        """
        if self._raster is None:
            logger.warning("Raster not opened, defaulting to water")
            return True
        
        try:
            # Convert to pixel coordinates
            row, col = self._raster.index(lon, lat)
            
            # Calculate buffer in pixels
            pixels_per_km = 1.0 / (self._resolution_deg * 111.0)
            buffer_pixels = int(self.buffer_km * pixels_per_km)
            
            # Define read window with buffer
            row_min = max(0, row - buffer_pixels)
            row_max = min(self._raster.height, row + buffer_pixels + 1)
            col_min = max(0, col - buffer_pixels)
            col_max = min(self._raster.width, col + buffer_pixels + 1)
            
            window = Window.from_slices(
                (row_min, row_max),
                (col_min, col_max)
            )
            
            # Read raster window
            data = self._raster.read(1, window=window)
            
            # Check center pixel
            center_row = row - row_min
            center_col = col - col_min
            
            if center_row < data.shape[0] and center_col < data.shape[1]:
                if data[center_row, center_col] >= self.threshold:
                    return True
            
            # Check if any nearby pixels are water
            if np.any(data >= self.threshold):
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Water check failed for ({lat}, {lon}): {e}")
            return True  # Default to water on errors
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


class ViolationDetector:
    """
    Detector for physical and environmental violations in vessel trajectories.
    
    Integrates speed checking, Kalman deviation analysis, and land detection
    to identify points exhibiting spoofing-like behavior.
    
    Usage:
        detector = ViolationDetector(raster_path="/path/to/gsw.vrt")
        violations = detector.analyze_route(route_points)
    """
    
    def __init__(self,
                 max_speed_knots: float = MAX_SHIP_SPEED_KNOTS,
                 deviation_threshold_km: float = 5.0,
                 small_jump_km: float = SMALL_JUMP_THRESHOLD_KM,
                 gap_threshold_minutes: float = 7.0,
                 raster_path: Optional[str] = None):
        """
        Initialize the violation detector.
        
        Args:
            max_speed_knots: Maximum plausible vessel speed
            deviation_threshold_km: Threshold for Kalman residuals
            small_jump_km: Minimum displacement to trigger violations
            gap_threshold_minutes: Kalman filter reset threshold
            raster_path: Optional path to GSW raster for land detection
        """
        self.max_speed_knots = max_speed_knots
        self.deviation_threshold_km = deviation_threshold_km
        self.small_jump_km = small_jump_km
        self.gap_threshold_minutes = gap_threshold_minutes
        
        # Initialize Kalman filter
        kalman_config = KalmanConfig(
            deviation_threshold_km=deviation_threshold_km,
            gap_threshold_minutes=gap_threshold_minutes
        )
        self.kalman = MaritimeKalmanFilter(kalman_config)
        
        # Initialize water checker if raster provided
        self.water_checker: Optional[WaterMaskChecker] = None
        if raster_path:
            try:
                self.water_checker = WaterMaskChecker(raster_path)
                self.water_checker.open()
            except Exception as e:
                logger.warning(f"Water checker unavailable: {e}")
                self.water_checker = None
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points.
        
        Args:
            lat1, lon1: First point in degrees
            lat2, lon2: Second point in degrees
            
        Returns:
            Distance in kilometers
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        
        # Handle longitude wraparound
        dlon = lon2 - lon1
        if dlon > 180:
            dlon -= 360
        elif dlon < -180:
            dlon += 360
        dlon_rad = math.radians(dlon)
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon_rad / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        return EARTH_RADIUS_KM * c
    
    def _check_speed_violation(self,
                               prev_point: AISPoint,
                               curr_point: AISPoint,
                               displacement_km: float) -> Optional[float]:
        """
        Check for speed violation between two points.
        
        Args:
            prev_point: Previous AIS point
            curr_point: Current AIS point
            displacement_km: Distance between points in km
            
        Returns:
            Speed in knots if violation, else None
        """
        dt_hours = (curr_point.timestamp - prev_point.timestamp).total_seconds() / 3600.0
        if dt_hours <= 0:
            return None
        
        speed_kmh = displacement_km / dt_hours
        speed_knots = speed_kmh / 1.852
        
        if speed_knots > self.max_speed_knots:
            return speed_knots
        return None
    
    def _check_land_violation(self, point: AISPoint) -> bool:
        """
        Check if a point is on land.
        
        Args:
            point: AIS point to check
            
        Returns:
            True if point is on land (violation), False if on water
        """
        if self.water_checker is None:
            return False  # No raster, assume water
        
        return not self.water_checker.is_on_water(point.latitude, point.longitude)
    
    def analyze_route(self, route: List[AISPoint]) -> List[ViolationRecord]:
        """
        Analyze a vessel route for violations.
        
        Processes the route sequentially, maintaining Kalman filter state
        and checking each point for speed, deviation, and land violations.
        
        Args:
            route: List of AISPoint objects sorted by timestamp
            
        Returns:
            List of ViolationRecord instances
        """
        if len(route) < 2:
            return []
        
        violations: List[ViolationRecord] = []
        
        # Sort by timestamp
        sorted_route = sorted(route, key=lambda p: p.timestamp)
        
        # Initialize Kalman filter
        self.kalman.initialize(sorted_route[0], sorted_route[1])
        
        prev_point = sorted_route[0]
        
        for i, point in enumerate(sorted_route[1:], start=1):
            # Calculate displacement from previous point
            displacement_km = self.haversine_distance(
                prev_point.latitude, prev_point.longitude,
                point.latitude, point.longitude
            )
            
            suppress_small = displacement_km < self.small_jump_km
            
            # Check for gap requiring Kalman reset
            gap_minutes = (point.timestamp - prev_point.timestamp).total_seconds() / 60.0
            allow_deviation = True
            
            if gap_minutes > self.gap_threshold_minutes:
                # Reset Kalman filter
                if i + 1 < len(sorted_route):
                    self.kalman.initialize(point, sorted_route[i + 1])
                else:
                    self.kalman.initialize(point)
                allow_deviation = False
                
                # Still check speed across the gap
                speed = self._check_speed_violation(prev_point, point, displacement_km)
                if speed is not None and not suppress_small:
                    violations.append(ViolationRecord(
                        index=i,
                        timestamp=point.timestamp.isoformat(),
                        latitude=point.latitude,
                        longitude=point.longitude,
                        violation_type=ViolationType.SPEED,
                        value=speed,
                        threshold=self.max_speed_knots
                    ))
            else:
                # Normal processing with Kalman filter
                
                # Check land violation
                if self._check_land_violation(point) and not suppress_small:
                    violations.append(ViolationRecord(
                        index=i,
                        timestamp=point.timestamp.isoformat(),
                        latitude=point.latitude,
                        longitude=point.longitude,
                        violation_type=ViolationType.LAND,
                        value=1.0,
                        threshold=0.0
                    ))
                
                # Check deviation violation
                if allow_deviation:
                    deviation_km = self.kalman.update(point)
                    
                    if deviation_km > self.deviation_threshold_km and not suppress_small:
                        violations.append(ViolationRecord(
                            index=i,
                            timestamp=point.timestamp.isoformat(),
                            latitude=point.latitude,
                            longitude=point.longitude,
                            violation_type=ViolationType.DEVIATION,
                            value=deviation_km,
                            threshold=self.deviation_threshold_km
                        ))
                
                # Check speed violation
                speed = self._check_speed_violation(prev_point, point, displacement_km)
                if speed is not None and not suppress_small:
                    violations.append(ViolationRecord(
                        index=i,
                        timestamp=point.timestamp.isoformat(),
                        latitude=point.latitude,
                        longitude=point.longitude,
                        violation_type=ViolationType.SPEED,
                        value=speed,
                        threshold=self.max_speed_knots
                    ))
            
            prev_point = point
        
        return violations
    
    def close(self) -> None:
        """Clean up resources."""
        if self.water_checker is not None:
            self.water_checker.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def create_violation_detector(
    max_speed_knots: float = MAX_SHIP_SPEED_KNOTS,
    deviation_threshold_km: float = 5.0,
    raster_path: Optional[str] = None
) -> ViolationDetector:
    """
    Factory function for creating a violation detector.
    
    Args:
        max_speed_knots: Maximum plausible vessel speed
        deviation_threshold_km: Kalman deviation threshold
        raster_path: Optional path to GSW raster
        
    Returns:
        Configured ViolationDetector instance
    """
    return ViolationDetector(
        max_speed_knots=max_speed_knots,
        deviation_threshold_km=deviation_threshold_km,
        raster_path=raster_path
    )
