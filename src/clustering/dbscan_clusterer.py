#!/usr/bin/env python3
"""
DBSCAN-based Spoofing Zone Clustering

This module implements the second stage of the spoofing detection pipeline:
aggregating per-vessel spoofing episodes into regional clusters that indicate
persistent or coordinated interference zones.

The clustering process:
1. Extracts boundary points (last clean before, first clean after) from episodes
2. Constructs "jump lines" connecting boundary pairs
3. Finds spatial intersections of jump lines across vessels
4. Applies DBSCAN to cluster nearby intersections
5. Stabilizes clusters through outlier removal and persistence filtering

The resulting clusters represent geographic regions where multiple independent
vessels exhibited correlated spoofing behavior.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from datetime import datetime, timedelta
import itertools

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


# Physical constants
EARTH_RADIUS_KM = 6371.0


@dataclass
class BoundaryPoint:
    """
    A clean AIS point at the boundary of a spoofing episode.
    
    Attributes:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        timestamp: ISO timestamp string
        boundary_type: 'before' or 'after' indicating position relative to episode
        mmsi: Vessel identifier
        episode_idx: Index of the associated episode
    """
    latitude: float
    longitude: float
    timestamp: str
    boundary_type: str  # 'before' or 'after'
    mmsi: str
    episode_idx: int
    
    def to_tuple(self) -> Tuple[float, float]:
        """Return (lat, lon) tuple."""
        return (self.latitude, self.longitude)


@dataclass
class JumpLine:
    """
    A line segment connecting the before and after boundary points of an episode.
    
    The jump line represents the apparent displacement during spoofing,
    connecting where the vessel was last seen at its true position to
    where it reappeared after the episode.
    """
    start: BoundaryPoint
    end: BoundaryPoint
    mmsi: str
    episode_idx: int
    
    @property
    def start_time(self) -> Optional[datetime]:
        """Parse start timestamp."""
        try:
            return datetime.fromisoformat(self.start.timestamp.replace('Z', '+00:00'))
        except:
            return None
    
    @property
    def end_time(self) -> Optional[datetime]:
        """Parse end timestamp."""
        try:
            return datetime.fromisoformat(self.end.timestamp.replace('Z', '+00:00'))
        except:
            return None


@dataclass
class IntersectionPoint:
    """
    A point where two jump lines from different vessels intersect.
    
    Intersections indicate that two independent vessels experienced
    correlated spoofing, suggesting an external interference source.
    """
    latitude: float
    longitude: float
    line1_mmsi: str
    line2_mmsi: str
    line1_episode: int
    line2_episode: int
    
    def to_array(self) -> np.ndarray:
        """Return as numpy array [lat, lon]."""
        return np.array([self.latitude, self.longitude])


@dataclass
class SpoofingCluster:
    """
    A cluster of spoofing activity aggregated across multiple vessels.
    
    Attributes:
        cluster_id: Unique cluster identifier
        vessels: Set of MMSIs in this cluster
        centroid: (lat, lon) centroid of cluster
        intersections: List of intersection points
        boundary_points: List of boundary points from all vessels
        temporal_extent: (start, end) datetime range
    """
    cluster_id: int
    vessels: Set[str]
    centroid: Tuple[float, float]
    intersections: List[IntersectionPoint]
    boundary_points: List[BoundaryPoint]
    temporal_extent: Tuple[Optional[datetime], Optional[datetime]] = (None, None)
    
    @property
    def vessel_count(self) -> int:
        """Number of unique vessels in cluster."""
        return len(self.vessels)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cluster_id': self.cluster_id,
            'vessel_count': self.vessel_count,
            'vessels': list(self.vessels),
            'centroid_lat': self.centroid[0],
            'centroid_lon': self.centroid[1],
            'intersection_count': len(self.intersections),
            'temporal_start': self.temporal_extent[0].isoformat() if self.temporal_extent[0] else None,
            'temporal_end': self.temporal_extent[1].isoformat() if self.temporal_extent[1] else None
        }


@dataclass
class ClusteringConfig:
    """Configuration for DBSCAN clustering."""
    epsilon_km: float = 50.0          # Maximum distance for clustering
    min_samples: int = 5              # Minimum points to form cluster
    outlier_distance_factor: float = 3.0  # Factor for outlier removal
    min_vessels_per_cluster: int = 2  # Minimum vessels for valid cluster


class JumpLineAnalyzer:
    """
    Analyzes jump lines to find spatial intersections.
    
    Jump lines connect the boundary points of spoofing episodes.
    When jump lines from different vessels intersect geographically,
    it suggests correlated spoofing affecting multiple receivers.
    """
    
    def __init__(self):
        self._lines: List[JumpLine] = []
    
    def add_line(self, line: JumpLine) -> None:
        """Add a jump line to the analyzer."""
        self._lines.append(line)
    
    @staticmethod
    def _segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                           p3: Tuple[float, float], p4: Tuple[float, float]
                           ) -> Optional[Tuple[float, float]]:
        """
        Check if two line segments intersect and return intersection point.
        
        Uses parametric line intersection with segment bounds checking.
        
        Args:
            p1, p2: Endpoints of first segment
            p3, p4: Endpoints of second segment
            
        Returns:
            (lat, lon) of intersection or None if segments don't intersect
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Parallel or coincident
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within both segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        
        return None
    
    def find_intersections(self) -> List[IntersectionPoint]:
        """
        Find all pairwise intersections between jump lines.
        
        Only considers intersections between lines from different vessels.
        
        Returns:
            List of IntersectionPoint instances
        """
        intersections = []
        
        # Check all pairs of lines
        for i, line1 in enumerate(self._lines):
            for line2 in self._lines[i+1:]:
                # Skip same vessel
                if line1.mmsi == line2.mmsi:
                    continue
                
                # Get segment endpoints
                p1 = line1.start.to_tuple()
                p2 = line1.end.to_tuple()
                p3 = line2.start.to_tuple()
                p4 = line2.end.to_tuple()
                
                # Check for intersection
                intersection = self._segments_intersect(p1, p2, p3, p4)
                if intersection is not None:
                    intersections.append(IntersectionPoint(
                        latitude=intersection[0],
                        longitude=intersection[1],
                        line1_mmsi=line1.mmsi,
                        line2_mmsi=line2.mmsi,
                        line1_episode=line1.episode_idx,
                        line2_episode=line2.episode_idx
                    ))
        
        logger.info(f"Found {len(intersections)} intersections from {len(self._lines)} jump lines")
        return intersections
    
    def clear(self) -> None:
        """Clear all stored jump lines."""
        self._lines.clear()


class SpoofingClusterer:
    """
    DBSCAN-based clustering for spoofing zone identification.
    
    Clusters intersection points (or boundary point centroids) to identify
    geographic regions with correlated spoofing activity across multiple vessels.
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize the clusterer.
        
        Args:
            config: ClusteringConfig instance or None for defaults
        """
        self.config = config or ClusteringConfig()
        self._dbscan: Optional[DBSCAN] = None
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in kilometers."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        return EARTH_RADIUS_KM * c
    
    def cluster_intersections(self, 
                             intersections: List[IntersectionPoint]
                             ) -> Tuple[Dict[int, List[IntersectionPoint]], List[IntersectionPoint]]:
        """
        Cluster intersection points using DBSCAN with haversine metric.
        
        Args:
            intersections: List of intersection points to cluster
            
        Returns:
            Tuple of (clustered_points dict, noise_points list)
        """
        if not intersections:
            return {}, []
        
        # Convert to radians for DBSCAN
        coords = np.array([[p.latitude, p.longitude] for p in intersections])
        coords_rad = np.radians(coords)
        
        # Epsilon in radians
        epsilon_rad = self.config.epsilon_km / EARTH_RADIUS_KM
        
        # Run DBSCAN
        self._dbscan = DBSCAN(
            eps=epsilon_rad,
            min_samples=self.config.min_samples,
            metric='haversine'
        )
        labels = self._dbscan.fit_predict(coords_rad)
        
        # Organize results
        clustered: Dict[int, List[IntersectionPoint]] = defaultdict(list)
        noise: List[IntersectionPoint] = []
        
        for point, label in zip(intersections, labels):
            if label == -1:
                noise.append(point)
            else:
                clustered[label].append(point)
        
        logger.info(f"DBSCAN found {len(clustered)} clusters, {len(noise)} noise points")
        return dict(clustered), noise
    
    def cluster_boundary_centroids(self,
                                   vessel_data: Dict[str, List[BoundaryPoint]]
                                   ) -> Tuple[Dict[int, List[str]], List[str]]:
        """
        Alternative clustering using spoofed segment centroids per vessel.
        
        Instead of clustering intersection points, this clusters the centroid
        of each vessel's spoofed points directly.
        
        Args:
            vessel_data: Dict mapping MMSI to list of boundary points
            
        Returns:
            Tuple of (cluster_to_vessels dict, self_spoofer_list)
        """
        if not vessel_data:
            return {}, []
        
        # Compute centroid for each vessel's spoofed points
        vessel_centroids = []
        vessel_ids = []
        
        for mmsi, points in vessel_data.items():
            if not points:
                continue
            lats = [p.latitude for p in points]
            lons = [p.longitude for p in points]
            centroid = (np.mean(lats), np.mean(lons))
            vessel_centroids.append(centroid)
            vessel_ids.append(mmsi)
        
        if not vessel_centroids:
            return {}, []
        
        # Convert to radians
        coords = np.array(vessel_centroids)
        coords_rad = np.radians(coords)
        
        epsilon_rad = self.config.epsilon_km / EARTH_RADIUS_KM
        
        self._dbscan = DBSCAN(
            eps=epsilon_rad,
            min_samples=self.config.min_samples,
            metric='haversine'
        )
        labels = self._dbscan.fit_predict(coords_rad)
        
        # Organize results
        clustered: Dict[int, List[str]] = defaultdict(list)
        noise: List[str] = []
        
        for mmsi, label in zip(vessel_ids, labels):
            if label == -1:
                noise.append(mmsi)
            else:
                clustered[label].append(mmsi)
        
        return dict(clustered), noise


class ClusterStabilizer:
    """
    Post-processing to stabilize and filter spoofing clusters.
    
    Applies:
    1. Outlier removal (points far from cluster centroid)
    2. Minimum vessel count filtering
    3. Temporal extent calculation
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
    
    @staticmethod
    def _compute_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Compute geographic centroid of points."""
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        return (np.mean(lats), np.mean(lons))
    
    @staticmethod
    def _distance_to_centroid(point: Tuple[float, float],
                              centroid: Tuple[float, float]) -> float:
        """Calculate distance from point to centroid in km."""
        lat1, lon1 = point
        lat2, lon2 = centroid
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        return EARTH_RADIUS_KM * c
    
    def remove_outliers(self,
                       points: List[IntersectionPoint]
                       ) -> List[IntersectionPoint]:
        """
        Remove outlier points far from cluster centroid.
        
        Args:
            points: List of intersection points in a cluster
            
        Returns:
            Filtered list with outliers removed
        """
        if len(points) < 3:
            return points
        
        coords = [(p.latitude, p.longitude) for p in points]
        centroid = self._compute_centroid(coords)
        
        # Calculate distances to centroid
        distances = [self._distance_to_centroid(c, centroid) for c in coords]
        mean_dist = np.mean(distances)
        threshold = mean_dist * self.config.outlier_distance_factor
        
        # Filter outliers
        filtered = [p for p, d in zip(points, distances) if d <= threshold]
        
        if len(filtered) < len(points):
            logger.debug(f"Removed {len(points) - len(filtered)} outliers from cluster")
        
        return filtered
    
    def build_cluster(self,
                     cluster_id: int,
                     intersections: List[IntersectionPoint],
                     boundary_points: Optional[List[BoundaryPoint]] = None
                     ) -> Optional[SpoofingCluster]:
        """
        Build a finalized SpoofingCluster from intersection points.
        
        Args:
            cluster_id: Unique identifier for the cluster
            intersections: List of intersection points after outlier removal
            boundary_points: Optional list of boundary points for temporal extent
            
        Returns:
            SpoofingCluster instance or None if cluster doesn't meet criteria
        """
        if not intersections:
            return None
        
        # Extract unique vessels
        vessels: Set[str] = set()
        for p in intersections:
            vessels.add(p.line1_mmsi)
            vessels.add(p.line2_mmsi)
        
        # Check minimum vessel count
        if len(vessels) < self.config.min_vessels_per_cluster:
            logger.debug(f"Cluster {cluster_id} has only {len(vessels)} vessels, skipping")
            return None
        
        # Compute centroid
        coords = [(p.latitude, p.longitude) for p in intersections]
        centroid = self._compute_centroid(coords)
        
        # Calculate temporal extent from boundary points
        temporal_start = None
        temporal_end = None
        
        if boundary_points:
            timestamps = []
            for bp in boundary_points:
                try:
                    ts = datetime.fromisoformat(bp.timestamp.replace('Z', '+00:00'))
                    timestamps.append(ts)
                except:
                    pass
            if timestamps:
                temporal_start = min(timestamps)
                temporal_end = max(timestamps)
        
        return SpoofingCluster(
            cluster_id=cluster_id,
            vessels=vessels,
            centroid=centroid,
            intersections=intersections,
            boundary_points=boundary_points or [],
            temporal_extent=(temporal_start, temporal_end)
        )


def run_clustering_pipeline(
    episode_data: Dict[str, List[Dict]],
    route_data: Dict[str, List[Dict]],
    config: Optional[ClusteringConfig] = None
) -> Tuple[List[SpoofingCluster], List[str]]:
    """
    Run the complete clustering pipeline.
    
    Args:
        episode_data: Dict mapping MMSI to list of episode dictionaries
        route_data: Dict mapping MMSI to route point lists
        config: Optional clustering configuration
        
    Returns:
        Tuple of (list of SpoofingCluster, list of self-spoofer MMSIs)
    """
    config = config or ClusteringConfig()
    
    # Extract boundary points and build jump lines
    analyzer = JumpLineAnalyzer()
    vessel_boundaries: Dict[str, List[BoundaryPoint]] = defaultdict(list)
    
    for mmsi, episodes in episode_data.items():
        route = route_data.get(mmsi, [])
        if not route:
            continue
        
        for ep_idx, episode in enumerate(episodes):
            start_idx = episode.get('start_idx', 0)
            end_idx = episode.get('end_idx', len(route) - 1)
            
            # Extract boundary points
            before_point = None
            after_point = None
            
            if start_idx > 0 and start_idx - 1 < len(route):
                pt = route[start_idx - 1]
                before_point = BoundaryPoint(
                    latitude=pt.get('latitude', pt.get('lat', 0)),
                    longitude=pt.get('longitude', pt.get('lon', 0)),
                    timestamp=pt.get('timestamp', ''),
                    boundary_type='before',
                    mmsi=mmsi,
                    episode_idx=ep_idx
                )
                vessel_boundaries[mmsi].append(before_point)
            
            if end_idx + 1 < len(route):
                pt = route[end_idx + 1]
                after_point = BoundaryPoint(
                    latitude=pt.get('latitude', pt.get('lat', 0)),
                    longitude=pt.get('longitude', pt.get('lon', 0)),
                    timestamp=pt.get('timestamp', ''),
                    boundary_type='after',
                    mmsi=mmsi,
                    episode_idx=ep_idx
                )
                vessel_boundaries[mmsi].append(after_point)
            
            # Build jump line if both boundaries exist
            if before_point and after_point:
                analyzer.add_line(JumpLine(
                    start=before_point,
                    end=after_point,
                    mmsi=mmsi,
                    episode_idx=ep_idx
                ))
    
    # Find intersections
    intersections = analyzer.find_intersections()
    
    # Cluster intersections
    clusterer = SpoofingClusterer(config)
    clustered, noise_points = clusterer.cluster_intersections(intersections)
    
    # Stabilize clusters
    stabilizer = ClusterStabilizer(config)
    clusters = []
    
    for cluster_id, points in clustered.items():
        # Remove outliers
        filtered = stabilizer.remove_outliers(points)
        
        # Get boundary points for vessels in this cluster
        cluster_vessels = set()
        for p in filtered:
            cluster_vessels.add(p.line1_mmsi)
            cluster_vessels.add(p.line2_mmsi)
        
        cluster_boundaries = []
        for mmsi in cluster_vessels:
            cluster_boundaries.extend(vessel_boundaries.get(mmsi, []))
        
        # Build cluster
        cluster = stabilizer.build_cluster(cluster_id, filtered, cluster_boundaries)
        if cluster:
            clusters.append(cluster)
    
    # Identify self-spoofers (vessels not in any cluster)
    clustered_vessels = set()
    for c in clusters:
        clustered_vessels.update(c.vessels)
    
    all_vessels = set(episode_data.keys())
    self_spoofers = list(all_vessels - clustered_vessels)
    
    logger.info(f"Pipeline complete: {len(clusters)} clusters, {len(self_spoofers)} self-spoofers")
    
    return clusters, self_spoofers
