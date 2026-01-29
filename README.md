# Maritime GPS Spoofing Detection Framework

A motion-aware, marine-specific framework for detecting and characterizing GPS spoofing anomalies in global maritime positioning data using Automatic Identification System (AIS) telemetry.

## Overview

This repository contains the implementation of a two-stage detection pipeline for identifying GPS spoofing events in maritime vessel traffic:

1. **Stage 1 (Per-Vessel Anomaly Detection)**: Kalman-based trajectory prediction with finite-state machine (FSM) temporal grouping to detect physically implausible vessel motion
2. **Stage 2 (Spoofing-Zone Clustering)**: DBSCAN-based spatial clustering to aggregate correlated anomalies across multiple vessels and identify persistent spoofing hotspots

The framework processes AIS data containing vessel positions, speeds, and courses to flag spoofing episodes characterized by:
- Deviation from predicted trajectories (>5km Kalman residuals)
- Implausible speeds (>60 knots)
- Positions over land (using Global Surface Water raster data)

## Repository Structure

```
maritime_spoofing_repo/
├── src/
│   ├── preprocessing/          # AIS data ingestion and cleaning
│   │   ├── ais_parser.py
│   │   ├── route_merger.py
│   │   └── data_validator.py
│   ├── detection/              # Stage 1: Per-vessel spoofing detection
│   │   ├── kalman_filter.py
│   │   ├── violation_detector.py
│   │   ├── fsm_grouper.py
│   │   └── batch_processor.py
│   ├── clustering/             # Stage 2: Multi-vessel clustering
│   │   ├── boundary_extractor.py
│   │   ├── jump_line_analyzer.py
│   │   ├── dbscan_clusterer.py
│   │   └── cluster_stabilizer.py
│   ├── analysis/               # Post-processing and statistics
│   │   ├── temporal_patterns.py
│   │   ├── geometric_classifier.py
│   │   ├── sanctions_matcher.py
│   │   └── validation_metrics.py
│   └── visualization/          # Mapping and plotting utilities
│       ├── hotspot_mapper.py
│       ├── trajectory_plotter.py
│       └── timeline_generator.py
├── data/
│   ├── sample/                 # Sample data schemas (no actual data)
│   └── schemas/                # Data format specifications
├── config/
│   ├── detection_params.yaml   # Detection threshold configuration
│   └── clustering_params.yaml  # Clustering parameters
├── tests/                      # Unit tests (requires test fixtures)
└── docs/                       # Additional documentation
```

## Requirements

### System Dependencies

- Python 3.9+
- GDAL/OGR libraries (for raster processing)
- SQLite 3.35+ (for WAL mode support)

### Python Dependencies

See `requirements.txt` for the full dependency list. Core packages include:

- `numpy>=1.21.0`
- `scipy>=1.7.0`
- `filterpy>=1.4.5` (Kalman filter implementation)
- `scikit-learn>=1.0.0` (DBSCAN clustering)
- `rasterio>=1.2.0` (geospatial raster I/O)
- `cartopy>=0.20.0` (map projections)
- `folium>=0.12.0` (interactive mapping)

### Data Dependencies

**Important**: This repository requires external data sources that are not included:

1. **AIS Data**: Global satellite AIS telemetry in JSON format, organized by vessel MMSI
2. **Water Occurrence Raster**: Global Surface Water Occurrence dataset (JRC, ~40GB VRT mosaic)
3. **Sanctions Lists**: OFAC/EU/UK vessel sanctions databases for cross-referencing

The data directory contains only schema definitions and sample format specifications.

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd maritime_spoofing_repo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install GDAL (platform-specific)
# On macOS: brew install gdal
# On Ubuntu: sudo apt-get install libgdal-dev
```

## Configuration

Before running the pipeline, configure the following in `config/detection_params.yaml`:

```yaml
# Detection thresholds (defaults shown)
deviation_threshold_km: 5.0      # Kalman residual threshold
max_speed_knots: 60.0            # Maximum plausible vessel speed
docked_radius_km: 0.5            # Stationary vessel filter radius
coastal_buffer_km: 1.0           # Land detection buffer

# FSM parameters
start_window_minutes: 30         # Sliding window for quorum detection
start_quorum_ratio: 0.70         # Minimum bad point ratio to trigger
end_clean_minutes: 120           # Consecutive clean minutes to end event
min_event_duration_minutes: 30   # Minimum event duration filter

# Kalman filter parameters
process_noise: 1e-5              # Q matrix diagonal
measurement_noise: 1e-4          # R matrix diagonal
gap_threshold_minutes: 7         # Filter reset threshold for large gaps
```

## Usage

### Stage 1: Per-Vessel Detection

```bash
# Process a single month of AIS data
python -m src.detection.batch_processor \
    --input-dir /path/to/ais/data \
    --output-dir /path/to/results \
    --raster-path /path/to/gsw_occurrence.vrt \
    --workers 4

# Output: {vessel_id}_kalman_results.json.gz per input file
```

### Stage 2: Clustering

```bash
# Cluster detected spoofing episodes
python -m src.clustering.dbscan_clusterer \
    --input-dir /path/to/stage1/results \
    --output-dir /path/to/clusters \
    --epsilon-km 50.0 \
    --min-cluster-size 5

# Output: cluster_assignments.json, self_spoofers.json
```

### Analysis

```bash
# Generate temporal persistence classifications
python -m src.analysis.temporal_patterns \
    --cluster-file /path/to/cluster_assignments.json \
    --output-dir /path/to/analysis

# Generate geometric pattern classifications
python -m src.analysis.geometric_classifier \
    --cluster-file /path/to/cluster_assignments.json \
    --output-dir /path/to/analysis
```

## Methodology Notes

### Kalman Filter Formulation

The detection stage uses a constant-velocity Kalman filter for trajectory prediction:

- State vector: `x = [latitude, longitude, d_lat/dt, d_lon/dt]^T`
- Measurement vector: `z = [latitude, longitude]^T`
- Process noise Q and measurement noise R tuned for maritime GPS uncertainty

The filter is reset after gaps exceeding `gap_threshold_minutes` to avoid prediction drift.

### FSM State Logic

The finite-state machine implements a "start-by-quorum / end-by-consecutive" pattern:

- **IDLE**: Default state, monitoring for anomalies
- **CANDIDATE**: Transitional state when anomalies begin accumulating
- **ACTIVE**: Confirmed spoofing episode (>=70% bad points in 30-min window)

Events terminate after 120 consecutive clean minutes.

### Clustering Approach

DBSCAN is applied to jump-line intersection points with:
- Epsilon = 50km (matching AIS communication range)
- MinPts = 5 (requiring spatial agreement across multiple vessels)

Clusters are stabilized through outlier removal (3x mean intra-cluster distance) and temporal persistence filtering.

## Validation

The detection pipeline was validated against 22 major global ports (see paper Section 3.6). False positive rates remained below 1% in all validation regions, with residual false positives attributed to:
- MMSI reuse (multiple vessels sharing identifiers)
- Faulty transponders
- Transient GPS/AIS sensor anomalies

## Known Limitations

1. **Stealthy spoofing**: Physically consistent false trajectories may evade detection
2. **Self-spoofing ambiguity**: Cannot always distinguish self-spoofing from external interference when co-located
3. **Regional coverage gaps**: Detection sensitivity varies with AIS coverage density

## Citation

If you use this code in your research, please cite:

```
[Citation information withheld for anonymous review]
```

## License

This code is provided for academic review purposes. See LICENSE file for terms.

## Contact

For questions regarding this submission, please contact the authors through the conference review system.
