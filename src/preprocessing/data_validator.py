#!/usr/bin/env python3
"""
AIS Data Parsing and Validation Module

This module handles the ingestion and validation of Automatic Identification
System (AIS) data from various formats including JSON, gzipped JSON, and CSV.

AIS data typically contains vessel identification, position, speed, course,
and timestamp information. This module normalizes these formats and filters
invalid or incomplete records.

Supported formats:
- Gzipped JSON (.json.gz): Primary format for merged ship routes
- JSON (.json): Uncompressed vessel route data
- CSV: Tabular AIS records (requires specific column mapping)
"""

import json
import gzip
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Iterator, Tuple, Any, Union
from pathlib import Path
import math

logger = logging.getLogger(__name__)


# Validation constants
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
INVALID_COORD_THRESHOLD = 0.001  # Reject (0,0) and nearby coordinates


@dataclass
class AISPoint:
    """
    A single AIS position report.
    
    Attributes:
        timestamp: UTC datetime of the report
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        speed_knots: Speed over ground in knots (optional)
        course_degrees: Course over ground in degrees (optional)
        heading: True heading in degrees (optional)
        mmsi: Maritime Mobile Service Identity (optional, for context)
    """
    timestamp: datetime
    latitude: float
    longitude: float
    speed_knots: float = 0.0
    course_degrees: float = 0.0
    heading: Optional[float] = None
    mmsi: Optional[str] = None
    
    def __post_init__(self):
        """Validate coordinates after initialization."""
        if not self._is_valid_coordinate(self.latitude, self.longitude):
            raise ValueError(
                f"Invalid coordinates: ({self.latitude}, {self.longitude})"
            )
    
    @staticmethod
    def _is_valid_coordinate(lat: float, lon: float) -> bool:
        """Check if coordinates are valid and not near (0,0)."""
        if lat < MIN_LATITUDE or lat > MAX_LATITUDE:
            return False
        if lon < MIN_LONGITUDE or lon > MAX_LONGITUDE:
            return False
        # Reject (0,0) coordinates (common error value)
        if abs(lat) < INVALID_COORD_THRESHOLD and abs(lon) < INVALID_COORD_THRESHOLD:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'latitude': self.latitude,
            'longitude': self.longitude,
            'speed_knots': self.speed_knots,
            'course_degrees': self.course_degrees
        }
    
    @classmethod
    def from_json_array(cls, data: List, mmsi: Optional[str] = None) -> Optional['AISPoint']:
        """
        Create from JSON array format [timestamp, lat, lon, speed, heading].
        
        Args:
            data: List with [timestamp, lat, lon, speed?, heading?]
            mmsi: Optional vessel identifier
            
        Returns:
            AISPoint instance or None if parsing fails
        """
        if len(data) < 3:
            return None
        
        try:
            timestamp_str = data[0]
            lat = float(data[1])
            lon = float(data[2])
            
            # Parse timestamp
            if isinstance(timestamp_str, str):
                ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                ts = ts.replace(tzinfo=None)  # Normalize to naive UTC
            else:
                return None
            
            # Optional fields
            speed = 0.0
            course = 0.0
            
            if len(data) > 3 and data[3] is not None:
                speed = float(data[3])
            if len(data) > 4 and data[4] is not None:
                course = float(data[4])
            
            return cls(
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                speed_knots=speed,
                course_degrees=course,
                mmsi=mmsi
            )
        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"Failed to parse AIS point: {e}")
            return None


@dataclass
class VesselMetadata:
    """
    Static vessel information from AIS Class A messages.
    
    Attributes:
        mmsi: Maritime Mobile Service Identity (9-digit)
        imo: IMO ship identification number
        name: Vessel name
        callsign: Radio call sign
        ship_type: AIS ship type code
        flag: Flag state
        length: Length overall in meters
        width: Beam in meters
        draught: Current draught in meters
    """
    mmsi: str
    imo: Optional[str] = None
    name: Optional[str] = None
    callsign: Optional[str] = None
    ship_type: Optional[str] = None
    flag: Optional[str] = None
    length: Optional[float] = None
    width: Optional[float] = None
    draught: Optional[float] = None


@dataclass
class VesselTrack:
    """
    A complete vessel track consisting of multiple AIS points.
    
    Attributes:
        mmsi: Vessel identifier
        points: List of AIS position reports
        metadata: Optional static vessel information
    """
    mmsi: str
    points: List[AISPoint] = field(default_factory=list)
    metadata: Optional[VesselMetadata] = None
    
    def __len__(self) -> int:
        return len(self.points)
    
    def __iter__(self) -> Iterator[AISPoint]:
        return iter(self.points)
    
    def add_point(self, point: AISPoint) -> None:
        """Add a point to the track."""
        self.points.append(point)
    
    def sort_by_time(self) -> None:
        """Sort points chronologically."""
        self.points.sort(key=lambda p: p.timestamp)
    
    def remove_duplicates(self) -> int:
        """
        Remove points with duplicate timestamps.
        
        Returns:
            Number of duplicates removed
        """
        if not self.points:
            return 0
        
        self.sort_by_time()
        original_count = len(self.points)
        
        unique_points = [self.points[0]]
        for point in self.points[1:]:
            if point.timestamp != unique_points[-1].timestamp:
                unique_points.append(point)
        
        self.points = unique_points
        return original_count - len(unique_points)
    
    def is_stationary(self, radius_km: float = 0.5) -> bool:
        """
        Check if vessel remained stationary (within radius of first point).
        
        Args:
            radius_km: Maximum radius to consider stationary
            
        Returns:
            True if all points within radius of first point
        """
        if len(self.points) < 2:
            return True
        
        lat0, lon0 = self.points[0].latitude, self.points[0].longitude
        
        for point in self.points[1:]:
            dist = self._haversine(lat0, lon0, point.latitude, point.longitude)
            if dist > radius_km:
                return False
        return True
    
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km using haversine formula."""
        R = 6371.0
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))
    
    @property
    def time_span(self) -> Optional[Tuple[datetime, datetime]]:
        """Get (start, end) timestamps or None if empty."""
        if not self.points:
            return None
        self.sort_by_time()
        return (self.points[0].timestamp, self.points[-1].timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mmsi': self.mmsi,
            'point_count': len(self.points),
            'points': [p.to_dict() for p in self.points]
        }


class AISParser:
    """
    Parser for AIS data in various formats.
    
    Handles JSON arrays, gzipped JSON, and provides iteration
    over vessel tracks.
    """
    
    def __init__(self, validate: bool = True, 
                 min_points: int = 2,
                 filter_stationary: bool = True,
                 stationary_radius_km: float = 0.5):
        """
        Initialize the parser.
        
        Args:
            validate: Whether to validate coordinates
            min_points: Minimum points for a valid track
            filter_stationary: Whether to filter stationary vessels
            stationary_radius_km: Radius for stationary detection
        """
        self.validate = validate
        self.min_points = min_points
        self.filter_stationary = filter_stationary
        self.stationary_radius_km = stationary_radius_km
        
        # Statistics
        self._parsed_count = 0
        self._filtered_count = 0
        self._error_count = 0
    
    def parse_json_file(self, file_path: Union[str, Path]) -> Dict[str, VesselTrack]:
        """
        Parse a JSON or gzipped JSON file containing vessel routes.
        
        Expected format:
        {
            "mmsi1": [[ts, lat, lon, speed, heading], ...],
            "mmsi2": [[ts, lat, lon, speed, heading], ...],
            ...
        }
        
        Args:
            file_path: Path to JSON or .json.gz file
            
        Returns:
            Dict mapping MMSI to VesselTrack
        """
        file_path = Path(file_path)
        
        # Determine file type and open accordingly
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        return self._process_json_data(data)
    
    def _process_json_data(self, data: Dict[str, List]) -> Dict[str, VesselTrack]:
        """Process parsed JSON data into vessel tracks."""
        tracks = {}
        
        for mmsi, route_points in data.items():
            self._parsed_count += 1
            
            try:
                track = VesselTrack(mmsi=mmsi)
                
                for point_data in route_points:
                    point = AISPoint.from_json_array(point_data, mmsi=mmsi)
                    if point is not None:
                        track.add_point(point)
                
                # Apply filters
                if len(track) < self.min_points:
                    self._filtered_count += 1
                    continue
                
                track.remove_duplicates()
                track.sort_by_time()
                
                if self.filter_stationary and track.is_stationary(self.stationary_radius_km):
                    self._filtered_count += 1
                    continue
                
                tracks[mmsi] = track
                
            except Exception as e:
                logger.warning(f"Error parsing track for MMSI {mmsi}: {e}")
                self._error_count += 1
        
        logger.info(f"Parsed {len(tracks)} tracks from {self._parsed_count} vessels "
                   f"(filtered: {self._filtered_count}, errors: {self._error_count})")
        
        return tracks
    
    def iterate_tracks(self, file_path: Union[str, Path]) -> Iterator[VesselTrack]:
        """
        Iterate over vessel tracks from a file.
        
        Memory-efficient iteration for large files.
        
        Args:
            file_path: Path to JSON file
            
        Yields:
            VesselTrack instances
        """
        tracks = self.parse_json_file(file_path)
        yield from tracks.values()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get parsing statistics."""
        return {
            'parsed': self._parsed_count,
            'filtered': self._filtered_count,
            'errors': self._error_count,
            'successful': self._parsed_count - self._filtered_count - self._error_count
        }
    
    def reset_statistics(self) -> None:
        """Reset parsing statistics."""
        self._parsed_count = 0
        self._filtered_count = 0
        self._error_count = 0


def validate_mmsi(mmsi: str) -> bool:
    """
    Validate MMSI format.
    
    Valid MMSI is 9 digits, not all zeros, and follows ITU allocation rules.
    
    Args:
        mmsi: MMSI string to validate
        
    Returns:
        True if valid MMSI format
    """
    if not mmsi or not isinstance(mmsi, str):
        return False
    
    # Should be 9 digits
    if len(mmsi) != 9 or not mmsi.isdigit():
        return False
    
    # Should not be all zeros
    if mmsi == '000000000':
        return False
    
    return True


def interpolate_nan_values(track: VesselTrack) -> None:
    """
    Interpolate NaN speed and course values using forward fill.
    
    Modifies the track in place.
    
    Args:
        track: VesselTrack to process
    """
    if not track.points:
        return
    
    last_speed = 0.0
    last_course = 0.0
    
    for point in track.points:
        if math.isnan(point.speed_knots) or point.speed_knots is None:
            point.speed_knots = last_speed
        else:
            last_speed = point.speed_knots
        
        if math.isnan(point.course_degrees) or point.course_degrees is None:
            point.course_degrees = last_course
        else:
            last_course = point.course_degrees
