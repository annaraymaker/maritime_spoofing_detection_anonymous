#!/usr/bin/env python3
"""
Batch Processor for Maritime Spoofing Detection

This module orchestrates the Stage 1 detection pipeline, processing
multiple AIS data files in parallel and generating per-vessel spoofing
episode results.

The processor:
1. Loads AIS data from gzipped JSON files
2. Applies per-vessel Kalman filtering and violation detection
3. Groups violations into episodes using the FSM
4. Outputs compressed results files

Usage:
    python -m src.detection.batch_processor --input-dir /path/to/ais \\
        --output-dir /path/to/results --raster-path /path/to/gsw.vrt
"""

import os
import sys
import glob
import gzip
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any

import yaml
from tqdm import tqdm

from .kalman_filter import MaritimeKalmanFilter, KalmanConfig
from .violation_detector import ViolationDetector, ViolationType
from .fsm_grouper import SpoofingFSM, FSMConfig, bin_violations_by_minute
from ..preprocessing.data_validator import AISParser, VesselTrack, interpolate_nan_values


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessorConfig:
    """Configuration container for batch processing."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Defaults
        self.deviation_threshold_km = 5.0
        self.max_speed_knots = 60.0
        self.docked_radius_km = 0.5
        self.gap_threshold_minutes = 7.0
        self.start_window_minutes = 30
        self.start_quorum_ratio = 0.70
        self.end_clean_minutes = 120
        self.min_event_duration_minutes = 30
        self.parallel_workers = 4
        
        if config_path:
            self._load_from_yaml(config_path)
    
    def _load_from_yaml(self, path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logger.info(f"Loaded configuration from {path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")


class VesselProcessor:
    """
    Processor for individual vessel tracks.
    
    Combines violation detection and FSM grouping to produce
    spoofing episode records.
    """
    
    def __init__(self, config: BatchProcessorConfig, raster_path: Optional[str] = None):
        self.config = config
        self.raster_path = raster_path
        
        # Initialize detector
        self.detector = ViolationDetector(
            max_speed_knots=config.max_speed_knots,
            deviation_threshold_km=config.deviation_threshold_km,
            gap_threshold_minutes=config.gap_threshold_minutes,
            raster_path=raster_path
        )
        
        # FSM configuration
        self.fsm_config = FSMConfig(
            start_window_minutes=config.start_window_minutes,
            start_quorum_ratio=config.start_quorum_ratio,
            end_clean_minutes=config.end_clean_minutes,
            min_event_duration_minutes=config.min_event_duration_minutes
        )
    
    def process_track(self, track: VesselTrack) -> Dict[str, Any]:
        """
        Process a single vessel track for spoofing episodes.
        
        Args:
            track: VesselTrack to analyze
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'mmsi': track.mmsi,
            'point_count': len(track),
            'violations': [],
            'episodes': [],
            'status': 'success',
            'skip_reason': None
        }
        
        # Validate track
        if len(track) < 2:
            result['status'] = 'skipped'
            result['skip_reason'] = 'short_route'
            return result
        
        if track.is_stationary(self.config.docked_radius_km):
            result['status'] = 'skipped'
            result['skip_reason'] = 'stationary'
            return result
        
        try:
            # Interpolate missing values
            interpolate_nan_values(track)
            
            # Detect violations
            violations = self.detector.analyze_route(track.points)
            result['violations'] = [v.to_dict() for v in violations]
            
            if not violations:
                result['status'] = 'clean'
                return result
            
            # Group into episodes using FSM
            fsm = SpoofingFSM(self.fsm_config)
            minute_records = bin_violations_by_minute(
                [v.to_dict() for v in violations],
                len(track)
            )
            
            for minute in minute_records:
                fsm.process_minute(minute)
            
            fsm.finalize()
            episodes = fsm.get_episodes()
            
            result['episodes'] = [ep.to_dict() for ep in episodes]
            
            if episodes:
                result['status'] = 'flagged'
            else:
                result['status'] = 'violations_only'
            
        except Exception as e:
            logger.error(f"Error processing MMSI {track.mmsi}: {e}")
            result['status'] = 'error'
            result['skip_reason'] = str(e)
        
        return result
    
    def close(self):
        """Release resources."""
        self.detector.close()


def process_file(
    file_path: str,
    output_dir: str,
    config: BatchProcessorConfig,
    raster_path: Optional[str] = None
) -> Tuple[int, int, int, Counter]:
    """
    Process a single AIS data file.
    
    Args:
        file_path: Path to input JSON file
        output_dir: Directory for output files
        config: Processing configuration
        raster_path: Optional path to GSW raster
        
    Returns:
        Tuple of (processed, flagged, skipped, skip_reasons)
    """
    filename = os.path.basename(file_path)
    logger.info(f"Processing {filename}")
    
    # Initialize processor
    processor = VesselProcessor(config, raster_path)
    parser = AISParser(
        min_points=2,
        filter_stationary=False,  # We handle this in the processor
        stationary_radius_km=config.docked_radius_km
    )
    
    processed = 0
    flagged = 0
    skipped = 0
    skip_reasons = Counter()
    
    try:
        # Parse input file
        tracks = parser.parse_json_file(file_path)
        
        # Prepare output files
        base_name = filename.replace('.json.gz', '').replace('.json', '')
        results_path = os.path.join(output_dir, f"{base_name}_results.json.gz")
        flagged_path = os.path.join(output_dir, f"{base_name}_flagged.json.gz")
        
        all_results = []
        flagged_results = []
        
        # Process each track
        for mmsi, track in tqdm(tracks.items(), desc=f"Processing {filename}"):
            result = processor.process_track(track)
            all_results.append(result)
            
            if result['status'] == 'skipped':
                skipped += 1
                if result['skip_reason']:
                    skip_reasons[result['skip_reason']] += 1
            elif result['status'] == 'error':
                skipped += 1
                skip_reasons['error'] += 1
            else:
                processed += 1
                if result['status'] == 'flagged':
                    flagged += 1
                    flagged_results.append(result)
        
        # Write results
        with gzip.open(results_path, 'wt', encoding='utf-8') as f:
            json.dump(all_results, f, separators=(',', ':'))
        
        with gzip.open(flagged_path, 'wt', encoding='utf-8') as f:
            json.dump(flagged_results, f, separators=(',', ':'))
        
        logger.info(f"{filename}: processed={processed}, flagged={flagged}, skipped={skipped}")
        
    except Exception as e:
        logger.error(f"Fatal error processing {filename}: {e}")
        
    finally:
        processor.close()
    
    return processed, flagged, skipped, skip_reasons


def run_batch_processing(
    input_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    raster_path: Optional[str] = None,
    workers: int = 4,
    file_pattern: str = "*.json.gz"
) -> Dict[str, Any]:
    """
    Run batch processing on multiple AIS files.
    
    Args:
        input_dir: Directory containing AIS data files
        output_dir: Directory for output files
        config_path: Optional path to configuration YAML
        raster_path: Optional path to GSW raster
        workers: Number of parallel workers
        file_pattern: Glob pattern for input files
        
    Returns:
        Dictionary with processing summary
    """
    start_time = time.time()
    
    # Load configuration
    config = BatchProcessorConfig(config_path)
    config.parallel_workers = workers
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find input files
    pattern = os.path.join(input_dir, file_pattern)
    files = sorted(glob.glob(pattern))
    
    logger.info(f"Found {len(files)} files matching {file_pattern}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using {workers} workers")
    
    # Process files in parallel
    total_processed = 0
    total_flagged = 0
    total_skipped = 0
    global_skip_reasons = Counter()
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_file, f, output_dir, config, raster_path
            ): f for f in files
        }
        
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                processed, flagged, skipped, skip_reasons = future.result()
                total_processed += processed
                total_flagged += flagged
                total_skipped += skipped
                global_skip_reasons.update(skip_reasons)
            except Exception as e:
                logger.error(f"Error in {os.path.basename(file_path)}: {e}")
    
    elapsed = time.time() - start_time
    
    summary = {
        'total_processed': total_processed,
        'total_flagged': total_flagged,
        'total_skipped': total_skipped,
        'skip_reasons': dict(global_skip_reasons),
        'elapsed_seconds': elapsed,
        'files_processed': len(files)
    }
    
    logger.info("Processing complete:")
    logger.info(f"  Total processed: {total_processed:,}")
    logger.info(f"  Total flagged: {total_flagged:,}")
    logger.info(f"  Total skipped: {total_skipped:,}")
    logger.info(f"  Elapsed time: {elapsed/60:.2f} minutes")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Batch processor for maritime spoofing detection'
    )
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Directory containing AIS data files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Directory for output files'
    )
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--raster-path', '-r',
        help='Path to Global Surface Water raster'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--pattern', '-p',
        default='*.json.gz',
        help='Glob pattern for input files (default: *.json.gz)'
    )
    
    args = parser.parse_args()
    
    run_batch_processing(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        raster_path=args.raster_path,
        workers=args.workers,
        file_pattern=args.pattern
    )


if __name__ == '__main__':
    main()
