#!/usr/bin/env python3
"""
Temporal Pattern Classification for Spoofing Clusters

This module analyzes the temporal dynamics of spoofing clusters to classify
them into persistence patterns:

1. Sustained: Continuous activity over extended periods (>7 days)
2. Recurrent: Periodic activity with regular inter-peak intervals
3. Intermittent: Irregular bursts of activity
4. Isolated: Single short-duration events

These classifications help characterize the operational nature of interference
zones and may indicate different spoofing motivations or sources.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class TemporalPattern(Enum):
    """Classification of temporal spoofing patterns."""
    SUSTAINED = auto()     # Continuous multi-day activity
    RECURRENT = auto()     # Periodic activity with regular intervals
    INTERMITTENT = auto()  # Irregular sporadic bursts
    ISOLATED = auto()      # Single short event


@dataclass
class TemporalClassificationConfig:
    """Configuration for temporal pattern classification."""
    # Sustained thresholds
    sustained_min_days: int = 7
    sustained_max_variation: float = 0.10
    
    # Recurrent thresholds
    recurrent_min_peaks: int = 3
    recurrent_max_spacing_cv: float = 0.30
    
    # Isolated thresholds
    isolated_max_duration_days: int = 3
    isolated_max_secondary_ratio: float = 0.25
    
    # Time resolution
    time_bin_minutes: int = 1


@dataclass
class ActivityTimeSeries:
    """
    Time series representation of spoofing activity.
    
    Stores minute-resolution counts of affected vessels over time.
    """
    timestamps: List[datetime]
    vessel_counts: List[int]
    cluster_id: int
    
    @property
    def duration_days(self) -> float:
        """Total duration of activity in days."""
        if not self.timestamps or len(self.timestamps) < 2:
            return 0.0
        return (self.timestamps[-1] - self.timestamps[0]).total_seconds() / 86400.0
    
    @property
    def max_count(self) -> int:
        """Maximum vessel count."""
        return max(self.vessel_counts) if self.vessel_counts else 0
    
    @property
    def mean_count(self) -> float:
        """Mean vessel count."""
        return np.mean(self.vessel_counts) if self.vessel_counts else 0.0


@dataclass
class TemporalClassification:
    """
    Result of temporal pattern classification.
    
    Attributes:
        cluster_id: Cluster identifier
        pattern: Classified temporal pattern
        confidence: Classification confidence (0-1)
        metrics: Dict of computed metrics used for classification
    """
    cluster_id: int
    pattern: TemporalPattern
    confidence: float
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cluster_id': self.cluster_id,
            'pattern': self.pattern.name.lower(),
            'confidence': round(self.confidence, 3),
            'metrics': self.metrics
        }


class TemporalPatternClassifier:
    """
    Classifier for temporal spoofing patterns.
    
    Analyzes time series of spoofing activity to assign one of four
    pattern categories to each cluster.
    """
    
    def __init__(self, config: Optional[TemporalClassificationConfig] = None):
        self.config = config or TemporalClassificationConfig()
    
    def build_time_series(self,
                         episodes: List[Dict[str, Any]],
                         cluster_id: int) -> ActivityTimeSeries:
        """
        Build a minute-resolution time series from episode data.
        
        Args:
            episodes: List of episode dictionaries with timestamps
            cluster_id: Cluster identifier
            
        Returns:
            ActivityTimeSeries instance
        """
        # Extract all timestamps
        all_minutes = set()
        
        for episode in episodes:
            try:
                start = datetime.fromisoformat(episode['start_timestamp'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(episode['end_timestamp'].replace('Z', '+00:00'))
                
                # Generate minute bins
                current = start.replace(second=0, microsecond=0)
                while current <= end:
                    all_minutes.add(current)
                    current += timedelta(minutes=self.config.time_bin_minutes)
            except (KeyError, ValueError):
                continue
        
        if not all_minutes:
            return ActivityTimeSeries([], [], cluster_id)
        
        # Sort and count vessels per minute
        sorted_minutes = sorted(all_minutes)
        
        # Count overlapping episodes per minute
        minute_counts = defaultdict(int)
        for episode in episodes:
            try:
                start = datetime.fromisoformat(episode['start_timestamp'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(episode['end_timestamp'].replace('Z', '+00:00'))
                start = start.replace(second=0, microsecond=0)
                
                current = start
                while current <= end:
                    minute_counts[current] += 1
                    current += timedelta(minutes=self.config.time_bin_minutes)
            except (KeyError, ValueError):
                continue
        
        timestamps = sorted_minutes
        counts = [minute_counts[t] for t in timestamps]
        
        return ActivityTimeSeries(timestamps, counts, cluster_id)
    
    def _detect_peaks(self, 
                     counts: List[int],
                     threshold_ratio: float = 0.5) -> List[int]:
        """
        Detect activity peaks in time series.
        
        Args:
            counts: Vessel count time series
            threshold_ratio: Minimum ratio of max to be considered peak
            
        Returns:
            List of peak indices
        """
        if not counts or len(counts) < 3:
            return []
        
        max_count = max(counts)
        threshold = max_count * threshold_ratio
        
        peaks = []
        for i in range(1, len(counts) - 1):
            if counts[i] >= threshold:
                if counts[i] >= counts[i-1] and counts[i] >= counts[i+1]:
                    peaks.append(i)
        
        return peaks
    
    def _compute_daily_variation(self,
                                ts: ActivityTimeSeries) -> float:
        """
        Compute coefficient of variation in daily vessel counts.
        
        Args:
            ts: Activity time series
            
        Returns:
            Coefficient of variation (std/mean)
        """
        if not ts.timestamps or ts.duration_days < 2:
            return float('inf')
        
        # Aggregate to daily counts
        daily_counts = defaultdict(int)
        for t, c in zip(ts.timestamps, ts.vessel_counts):
            day = t.date()
            daily_counts[day] = max(daily_counts[day], c)
        
        counts = list(daily_counts.values())
        if len(counts) < 2:
            return float('inf')
        
        mean = np.mean(counts)
        if mean == 0:
            return float('inf')
        
        return np.std(counts) / mean
    
    def _compute_peak_spacing_cv(self,
                                ts: ActivityTimeSeries,
                                peak_indices: List[int]) -> float:
        """
        Compute coefficient of variation in peak spacing.
        
        Args:
            ts: Activity time series
            peak_indices: Indices of detected peaks
            
        Returns:
            CV of inter-peak intervals
        """
        if len(peak_indices) < 2:
            return float('inf')
        
        # Compute inter-peak intervals in minutes
        intervals = []
        for i in range(1, len(peak_indices)):
            dt = (ts.timestamps[peak_indices[i]] - 
                  ts.timestamps[peak_indices[i-1]])
            intervals.append(dt.total_seconds() / 60.0)
        
        if not intervals:
            return float('inf')
        
        mean = np.mean(intervals)
        if mean == 0:
            return float('inf')
        
        return np.std(intervals) / mean
    
    def classify(self, ts: ActivityTimeSeries) -> TemporalClassification:
        """
        Classify the temporal pattern of a spoofing cluster.
        
        Args:
            ts: Activity time series for the cluster
            
        Returns:
            TemporalClassification with pattern and confidence
        """
        metrics = {
            'duration_days': ts.duration_days,
            'max_count': ts.max_count,
            'mean_count': ts.mean_count
        }
        
        # Handle empty or very short series
        if not ts.timestamps or ts.duration_days < 0.1:
            return TemporalClassification(
                cluster_id=ts.cluster_id,
                pattern=TemporalPattern.ISOLATED,
                confidence=1.0,
                metrics=metrics
            )
        
        # Detect peaks
        peaks = self._detect_peaks(ts.vessel_counts)
        metrics['peak_count'] = len(peaks)
        
        # Compute classification features
        daily_cv = self._compute_daily_variation(ts)
        metrics['daily_cv'] = round(daily_cv, 3) if daily_cv != float('inf') else None
        
        peak_spacing_cv = self._compute_peak_spacing_cv(ts, peaks)
        metrics['peak_spacing_cv'] = round(peak_spacing_cv, 3) if peak_spacing_cv != float('inf') else None
        
        # Classification logic
        
        # Check for SUSTAINED pattern
        if (ts.duration_days >= self.config.sustained_min_days and
            daily_cv <= self.config.sustained_max_variation):
            confidence = min(1.0, ts.duration_days / 14.0)  # Higher for longer durations
            return TemporalClassification(
                ts.cluster_id, TemporalPattern.SUSTAINED, confidence, metrics
            )
        
        # Check for RECURRENT pattern
        if (len(peaks) >= self.config.recurrent_min_peaks and
            peak_spacing_cv <= self.config.recurrent_max_spacing_cv):
            confidence = min(1.0, len(peaks) / 5.0)
            return TemporalClassification(
                ts.cluster_id, TemporalPattern.RECURRENT, confidence, metrics
            )
        
        # Check for ISOLATED pattern
        if ts.duration_days <= self.config.isolated_max_duration_days:
            # Check for secondary peaks
            if len(peaks) > 1:
                sorted_counts = sorted([ts.vessel_counts[p] for p in peaks], reverse=True)
                if sorted_counts[1] / sorted_counts[0] > self.config.isolated_max_secondary_ratio:
                    # Multiple significant peaks -> INTERMITTENT
                    return TemporalClassification(
                        ts.cluster_id, TemporalPattern.INTERMITTENT, 0.6, metrics
                    )
            
            confidence = 1.0 - (ts.duration_days / self.config.isolated_max_duration_days)
            return TemporalClassification(
                ts.cluster_id, TemporalPattern.ISOLATED, confidence, metrics
            )
        
        # Default to INTERMITTENT
        return TemporalClassification(
            ts.cluster_id, TemporalPattern.INTERMITTENT, 0.5, metrics
        )
    
    def classify_cluster(self,
                        cluster_id: int,
                        episodes: List[Dict[str, Any]]) -> TemporalClassification:
        """
        Classify temporal pattern for a cluster given its episodes.
        
        Args:
            cluster_id: Cluster identifier
            episodes: List of episode dictionaries
            
        Returns:
            TemporalClassification result
        """
        ts = self.build_time_series(episodes, cluster_id)
        return self.classify(ts)


def classify_all_clusters(
    cluster_episodes: Dict[int, List[Dict[str, Any]]],
    config: Optional[TemporalClassificationConfig] = None
) -> Dict[int, TemporalClassification]:
    """
    Classify temporal patterns for all clusters.
    
    Args:
        cluster_episodes: Dict mapping cluster_id to list of episodes
        config: Optional classification configuration
        
    Returns:
        Dict mapping cluster_id to TemporalClassification
    """
    classifier = TemporalPatternClassifier(config)
    results = {}
    
    for cluster_id, episodes in cluster_episodes.items():
        results[cluster_id] = classifier.classify_cluster(cluster_id, episodes)
    
    # Log summary
    pattern_counts = defaultdict(int)
    for classification in results.values():
        pattern_counts[classification.pattern.name] += 1
    
    logger.info(f"Temporal pattern classification: {dict(pattern_counts)}")
    
    return results
