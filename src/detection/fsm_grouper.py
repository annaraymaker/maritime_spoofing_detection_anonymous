#!/usr/bin/env python3
"""
Finite State Machine for Temporal Grouping of Spoofing Episodes

This module implements the start-by-quorum / end-by-consecutive FSM pattern
for aggregating point-level anomalies into coherent spoofing episodes.

The FSM operates on minute-binned anomaly data and produces temporally bounded
event records with aggregated violation statistics.

State Machine Logic:
    IDLE -> CANDIDATE: First anomalies detected (< quorum)
    CANDIDATE -> ACTIVE: Quorum reached (>= 70% bad points in 30-min window)
    CANDIDATE -> IDLE: Anomalies dissipate before quorum
    ACTIVE -> IDLE: Clean gap end condition met (120 consecutive clean minutes)
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple, Any, Deque
import logging

logger = logging.getLogger(__name__)


class FSMState(Enum):
    """States of the spoofing detection FSM."""
    IDLE = auto()       # No active anomaly detection
    CANDIDATE = auto()  # Potential event building (below quorum)
    ACTIVE = auto()     # Confirmed spoofing episode


@dataclass
class MinuteRecord:
    """
    Aggregated violation data for a single minute bin.
    
    Attributes:
        timestamp: Floored minute timestamp
        point_indices: Indices of route points in this minute
        bad_points: Count of points with violations
        total_points: Total points in this minute
        reasons: Set of violation types present
        point_reasons: List of individual point reasons for aggregation
    """
    timestamp: datetime
    point_indices: List[int]
    bad_points: int
    total_points: int
    reasons: set = field(default_factory=set)
    point_reasons: List[str] = field(default_factory=list)
    
    @property
    def is_bad(self) -> bool:
        """Returns True if any violations occurred in this minute."""
        return len(self.reasons) > 0
    
    @property
    def bad_ratio(self) -> float:
        """Ratio of bad points to total points."""
        if self.total_points == 0:
            return 0.0
        return self.bad_points / self.total_points


@dataclass
class SpoofingEpisode:
    """
    A confirmed spoofing episode with temporal bounds and statistics.
    
    Attributes:
        start_idx: Index of first point in the episode
        end_idx: Index of last point in the episode
        start_timestamp: ISO timestamp of episode start
        end_timestamp: ISO timestamp of episode end
        duration_minutes: Episode duration
        point_count: Number of points in episode
        reasons: Aggregated violation reason counts
    """
    start_idx: int
    end_idx: int
    start_timestamp: str
    end_timestamp: str
    duration_minutes: float
    point_count: int
    reasons: str  # Serialized reason counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'start_timestamp': self.start_timestamp,
            'end_timestamp': self.end_timestamp,
            'duration_minutes': self.duration_minutes,
            'point_count': self.point_count,
            'reasons': self.reasons
        }


@dataclass
class FSMConfig:
    """Configuration parameters for the spoofing FSM."""
    start_window_minutes: int = 30       # Rolling window size
    start_quorum_ratio: float = 0.70     # Minimum bad ratio to activate
    end_clean_minutes: int = 120         # Clean minutes to deactivate
    min_event_duration_minutes: int = 30 # Minimum episode duration
    min_event_span_points: int = 1       # Minimum points per episode


class WindowEntry:
    """Entry in the sliding window deque."""
    def __init__(self, minute: MinuteRecord):
        self.timestamp = minute.timestamp
        self.is_bad = minute.is_bad
        self.reasons = minute.reasons
        self.bad_points = minute.bad_points
        self.total_points = minute.total_points
        self.indices = minute.point_indices
        self.point_reasons = minute.point_reasons


class SpoofingFSM:
    """
    Finite State Machine for detecting and grouping spoofing episodes.
    
    The FSM processes minute-binned violation data and emits bounded
    spoofing episodes when temporal persistence criteria are met.
    
    State Transitions:
        IDLE: Monitoring state. Transitions to CANDIDATE when anomalies appear.
        CANDIDATE: Building state. Transitions to ACTIVE if quorum is reached
                   within the rolling window, otherwise returns to IDLE.
        ACTIVE: Confirmed episode. Accumulates until a clean gap ends the event.
    
    The "start-by-quorum" rule requires a minimum fraction of bad points within
    a configurable time window. The "end-by-consecutive" rule requires a 
    configurable number of consecutive clean minutes to terminate an episode.
    """
    
    def __init__(self, config: Optional[FSMConfig] = None):
        """
        Initialize the FSM with the given configuration.
        
        Args:
            config: FSMConfig instance or None for defaults
        """
        self.config = config or FSMConfig()
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset all FSM state variables to initial values."""
        self.state = FSMState.IDLE
        
        # Sliding window for quorum detection
        self._window: Deque[WindowEntry] = deque()
        self._window_bad_points = 0
        self._window_total_points = 0
        
        # Active episode tracking
        self._event_start_time: Optional[datetime] = None
        self._event_start_idx: Optional[int] = None
        self._last_bad_time: Optional[datetime] = None
        self._last_bad_idx: Optional[int] = None
        self._clean_elapsed_minutes = 0.0
        self._event_reasons = Counter()
        
        # Previous minute tracking
        self._prev_minute: Optional[datetime] = None
        
        # Emitted episodes
        self._episodes: List[SpoofingEpisode] = []
    
    def _emit_episode(self) -> None:
        """
        Emit the current active episode and reset event tracking.
        
        Applies guardrails to filter spurious short events.
        """
        if (self._event_start_time is None or 
            self._last_bad_time is None or
            self._event_start_idx is None or 
            self._last_bad_idx is None):
            logger.warning("Cannot emit episode: missing boundary information")
            return
        
        duration = (self._last_bad_time - self._event_start_time).total_seconds() / 60.0
        span = self._last_bad_idx - self._event_start_idx + 1
        
        # Apply guardrails
        if duration < self.config.min_event_duration_minutes:
            logger.debug(f"Skipping episode: duration {duration:.1f}min < threshold")
            return
        if span < self.config.min_event_span_points:
            logger.debug(f"Skipping episode: span {span}pts < threshold")
            return
        
        # Format reasons summary
        reasons_str = ",".join(
            f"{reason}:{count}" 
            for reason, count in sorted(self._event_reasons.items())
        )
        
        episode = SpoofingEpisode(
            start_idx=self._event_start_idx,
            end_idx=self._last_bad_idx,
            start_timestamp=self._event_start_time.isoformat(),
            end_timestamp=self._last_bad_time.isoformat(),
            duration_minutes=duration,
            point_count=span,
            reasons=reasons_str
        )
        
        self._episodes.append(episode)
        logger.info(f"Episode emitted: indices {episode.start_idx}-{episode.end_idx}, "
                   f"duration {duration:.1f}min")
    
    def _clear_event_tracking(self) -> None:
        """Clear event tracking variables after emission or abandonment."""
        self._event_start_time = None
        self._event_start_idx = None
        self._last_bad_time = None
        self._last_bad_idx = None
        self._clean_elapsed_minutes = 0.0
        self._event_reasons = Counter()
    
    def _clear_window(self) -> None:
        """Clear the sliding window."""
        self._window.clear()
        self._window_bad_points = 0
        self._window_total_points = 0
    
    def _update_window(self, entry: WindowEntry) -> None:
        """
        Add a new entry to the sliding window and evict old entries.
        
        Args:
            entry: New WindowEntry to add
        """
        # Add new entry
        self._window.append(entry)
        self._window_bad_points += entry.bad_points
        self._window_total_points += entry.total_points
        
        # Evict entries outside the window
        window_threshold = timedelta(minutes=self.config.start_window_minutes)
        while self._window:
            oldest = self._window[0]
            age = entry.timestamp - oldest.timestamp
            if age > window_threshold:
                self._window.popleft()
                self._window_bad_points -= oldest.bad_points
                self._window_total_points -= oldest.total_points
            else:
                break
    
    def _get_window_span_minutes(self) -> int:
        """Get the time span of the current window in minutes (inclusive)."""
        if not self._window:
            return 0
        span = self._window[-1].timestamp - self._window[0].timestamp
        return int(span.total_seconds() / 60.0) + 1
    
    def _get_window_bad_ratio(self) -> float:
        """Get the ratio of bad points in the current window."""
        if self._window_total_points == 0:
            return 0.0
        return self._window_bad_points / self._window_total_points
    
    def _initialize_event_from_window(self) -> None:
        """Initialize event tracking from the current window contents."""
        earliest_bad_time = None
        earliest_bad_idx = None
        latest_bad_time = None
        latest_bad_idx = None
        
        for entry in self._window:
            if entry.is_bad and entry.indices:
                if earliest_bad_time is None:
                    earliest_bad_time = entry.timestamp
                    earliest_bad_idx = entry.indices[0]
                latest_bad_time = entry.timestamp
                latest_bad_idx = entry.indices[-1]
        
        # Use window boundaries as fallback
        if earliest_bad_time is None and self._window:
            earliest_bad_time = self._window[0].timestamp
            earliest_bad_idx = self._window[0].indices[0] if self._window[0].indices else 0
        if latest_bad_time is None and self._window:
            latest_bad_time = self._window[-1].timestamp
            latest_bad_idx = self._window[-1].indices[-1] if self._window[-1].indices else 0
        
        self._event_start_time = earliest_bad_time
        self._event_start_idx = earliest_bad_idx
        self._last_bad_time = latest_bad_time
        self._last_bad_idx = latest_bad_idx
        
        # Aggregate reasons from window
        self._event_reasons = Counter()
        for entry in self._window:
            self._event_reasons.update(entry.point_reasons)
        
        self._clean_elapsed_minutes = 0.0
    
    def process_minute(self, minute: MinuteRecord) -> Optional[SpoofingEpisode]:
        """
        Process a single minute of violation data through the FSM.
        
        This is the main entry point for the FSM. Call this method for each
        minute bin in chronological order.
        
        Args:
            minute: MinuteRecord containing aggregated violation data
            
        Returns:
            A SpoofingEpisode if one was completed by this minute, else None
        """
        entry = WindowEntry(minute)
        completed_episode = None
        
        # Calculate time since previous minute
        time_diff_minutes = 0.0
        if self._prev_minute is not None:
            time_diff_minutes = (minute.timestamp - self._prev_minute).total_seconds() / 60.0
        
        # Update sliding window
        self._update_window(entry)
        
        # Get window metrics
        window_span = self._get_window_span_minutes()
        bad_ratio = self._get_window_bad_ratio()
        has_full_coverage = window_span >= self.config.start_window_minutes
        ratio_ok = bad_ratio >= self.config.start_quorum_ratio
        
        # State machine logic
        if self.state == FSMState.IDLE:
            if has_full_coverage and ratio_ok:
                # Quorum reached - activate
                self.state = FSMState.ACTIVE
                self._initialize_event_from_window()
                logger.debug(f"IDLE -> ACTIVE at {minute.timestamp}, ratio={bad_ratio:.3f}")
        
        elif self.state == FSMState.ACTIVE:
            if minute.is_bad:
                # Update event extent
                self._last_bad_time = minute.timestamp
                if minute.point_indices:
                    self._last_bad_idx = minute.point_indices[-1]
                self._clean_elapsed_minutes = 0.0
                self._event_reasons.update(minute.point_reasons)
            else:
                # Accumulate clean time
                self._clean_elapsed_minutes += time_diff_minutes
                
                # Check for event termination
                if self._clean_elapsed_minutes >= self.config.end_clean_minutes:
                    self._emit_episode()
                    if self._episodes:
                        completed_episode = self._episodes[-1]
                    
                    # Return to IDLE
                    self.state = FSMState.IDLE
                    self._clear_event_tracking()
                    self._clear_window()
                    logger.debug(f"ACTIVE -> IDLE at {minute.timestamp}, "
                               f"clean gap {self._clean_elapsed_minutes:.1f}min")
        
        self._prev_minute = minute.timestamp
        return completed_episode
    
    def finalize(self) -> Optional[SpoofingEpisode]:
        """
        Finalize processing and emit any pending episode.
        
        Call this after all minutes have been processed to ensure
        any active episode is properly closed.
        
        Returns:
            A SpoofingEpisode if one was active, else None
        """
        completed_episode = None
        
        if self.state == FSMState.ACTIVE:
            self._emit_episode()
            if self._episodes:
                completed_episode = self._episodes[-1]
            self._clear_event_tracking()
            self._clear_window()
            self.state = FSMState.IDLE
        
        return completed_episode
    
    def get_episodes(self) -> List[SpoofingEpisode]:
        """
        Get all emitted episodes.
        
        Returns:
            List of SpoofingEpisode instances
        """
        return self._episodes.copy()
    
    def reset(self) -> None:
        """Reset the FSM for processing a new vessel track."""
        self._reset_state()


def bin_violations_by_minute(
    violations: List[Dict[str, Any]],
    route_length: int
) -> List[MinuteRecord]:
    """
    Bin point-level violations into minute records.
    
    Args:
        violations: List of violation dictionaries with 'index', 'timestamp', 'reason'
        route_length: Total number of points in the route
        
    Returns:
        List of MinuteRecord instances sorted by timestamp
    """
    minute_bins: Dict[datetime, MinuteRecord] = {}
    
    for v in violations:
        try:
            ts = datetime.fromisoformat(v['timestamp'].replace('Z', '+00:00'))
            ts = ts.replace(tzinfo=None)
        except (ValueError, KeyError):
            continue
        
        # Floor to minute
        minute_ts = ts.replace(second=0, microsecond=0)
        
        if minute_ts not in minute_bins:
            minute_bins[minute_ts] = MinuteRecord(
                timestamp=minute_ts,
                point_indices=[],
                bad_points=0,
                total_points=0
            )
        
        record = minute_bins[minute_ts]
        record.point_indices.append(v['index'])
        record.bad_points += 1
        record.total_points += 1
        record.reasons.add(v['reason'])
        record.point_reasons.append(v['reason'])
    
    # Sort by timestamp
    return sorted(minute_bins.values(), key=lambda r: r.timestamp)


def create_fsm(
    start_window_minutes: int = 30,
    start_quorum_ratio: float = 0.70,
    end_clean_minutes: int = 120,
    min_event_duration_minutes: int = 30
) -> SpoofingFSM:
    """
    Factory function to create a configured FSM instance.
    
    Args:
        start_window_minutes: Rolling window size for quorum detection
        start_quorum_ratio: Minimum bad point ratio to trigger activation
        end_clean_minutes: Consecutive clean minutes to terminate episode
        min_event_duration_minutes: Minimum episode duration for emission
        
    Returns:
        Configured SpoofingFSM instance
    """
    config = FSMConfig(
        start_window_minutes=start_window_minutes,
        start_quorum_ratio=start_quorum_ratio,
        end_clean_minutes=end_clean_minutes,
        min_event_duration_minutes=min_event_duration_minutes
    )
    return SpoofingFSM(config)
