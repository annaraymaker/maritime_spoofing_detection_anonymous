# Maritime GPS Spoofing Measurement Framework

A motion-aware, marine-specific framework for identifying and characterizing GPS spoofing anomalies in global maritime positioning data using Automatic Identification System (AIS) telemetry.

## Overview

This repository contains the implementation of a two-stage identification pipeline for identifying GPS spoofing events in maritime vessel traffic:

1. **Stage 1 (Per-Vessel Anomaly Identification)**: Kalman-based trajectory prediction with finite-state machine (FSM) temporal grouping to detect physically implausible vessel motion
2. **Stage 2 (Anomaly-Zone Clustering)**: DBSCAN-based spatial clustering to aggregate correlated anomalies across multiple vessels and identify persistent anomalous hotspots

The framework processes AIS data containing vessel positions, speeds, and courses to flag anomalous episodes characterized by:
- Deviation from predicted trajectories (>5km Kalman residuals)
- Implausible speeds (>60 knots)
- Positions over land (using Global Surface Water raster data)

A self-contained **NOAA reproduction module** (`noaa_validation/`) re-derives the
Gulf of Mexico spoofing event from NOAA's public AIS feed, independently of the
Spire dataset, so reviewers can run an end-to-end confirmation from open data on
a single daily file. See [NOAA Reproduction](#noaa-reproduction-open-data) below.

A second module (`dma_validation/`) runs the same pipeline over the Baltic /
Øresund region (Cluster 12) using the Danish Maritime Authority's public AIS
feed across a multi-day window. See [DMA Reproduction](#dma-reproduction-open-data) below.

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
├── noaa_validation/            # Open-data reproduction (Gulf of Mexico, Cluster 10)
│   ├── noaa_loader.py          #   schema-normalizing NOAA AIS loader
│   ├── anomaly_pipeline.py     #   Stage 1 (Kalman + violations + FSM)
│   ├── zone_analysis.py        #   single-region geometry + synchronization
│   ├── run_gulf_validation.py  #   one-command runner
│   ├── README.md               #   module-specific docs
│   └── (sample data, see below)
├── dma_validation/             # Open-data reproduction (Baltic/Oresund, Cluster 12)
│   ├── dma_loader.py           #   Danish Maritime Authority AIS loader (multi-day)
│   ├── anomaly_pipeline.py     #   Stage 1 (shared with noaa_validation)
│   ├── zone_analysis.py        #   geometry + synchronization (shared)
│   ├── run_baltic_validation.py#   multi-day runner
│   ├── download_dma.py         #   optional local DMA downloader
│   ├── README.md               #   module-specific docs
│   └── (synthetic sample, see below)
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

The `noaa_validation/` and `dma_validation/` modules are intentionally
lightweight and depend only on `numpy`, `pandas`, `matplotlib`, and `zstandard`
(the last only for `.zst` inputs; `dma_validation/download_dma.py` additionally
uses `requests`).

### Data Dependencies

**Important**: The main pipeline requires external data sources that are not included:

1. **AIS Data (primary study)**: Global satellite AIS telemetry (Spire Global) in JSON format, organized by vessel MMSI
2. **Water Occurrence Raster**: Global Surface Water Occurrence dataset (JRC, ~40GB VRT mosaic)
3. **Sanctions Lists**: OFAC/EU/UK vessel sanctions databases for cross-referencing

The `data/` directory contains only schema definitions and sample format specifications.

#### Data for the NOAA reproduction

The `noaa_validation/` module runs on **NOAA's public AIS feed** (no Spire access
required). Download daily files from the NOAA AIS data handler:

- Source: `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/index.html`
  (replace the year as needed).
- The reproduction in the paper uses **December 31, 2024** (`AIS_2024_12_31.zip`)
  as the event day and **January 1, 2025** as a control day.
- Each daily file is a single CSV (~750–840 MB uncompressed). The loader reads
  `.csv`, `.zip`, `.zst`, and `.gz` directly, so no manual extraction is needed.

NOAA distributes two CSV layouts in the wild; the loader handles **both**:

| Layout | Header | Datetime | Coord order |
|--------|--------|----------|-------------|
| Classic (marinecadastre) | `MMSI,BaseDateTime,LAT,LON,...` | `2024-12-31T00:00:08` | LAT, then LON |
| Lowercase variant | `mmsi,base_date_time,longitude,latitude,...` | `2025-01-01 00:00:00` | **LON, then LAT** |

A small **bundled sample** (`AIS_2024_12_31_SAMPLE.zip`, ~5 MB, 70 vessels) lets
reviewers run the module immediately without the full download; it reproduces the
same 10 confirmed-episode vessels and cross-shaped geometry as the full file.
A second sample (`sample_ais_variant_schema.csv`) exercises the lowercase layout.

> **Coverage note.** NOAA's feed is sourced from **U.S. terrestrial receivers**
> and therefore covers vessels in U.S. coastal waters **regardless of flag**
> (in the Gulf box, ~16% of vessels are foreign-flagged: Liberia, Marshall
> Islands, Panama, Bahamas, Norway, etc.). It is *not* a U.S.-flag-only feed.
> Counts are lower than the Spire (global satellite) figures because NOAA cannot
> see (a) spoofed endpoints that jump outside U.S. receiver range and (b) the
> broader fleet that satellite AIS captures globally — a coverage difference, not
> a flag-state restriction.

#### Data for the DMA reproduction

The `dma_validation/` module runs on the **Danish Maritime Authority (DMA)**
public AIS feed (no Spire access required). Download daily files from:

- Source: `http://aisdata.ais.dk/?prefix=`
- Files are named `aisdk-YYYY-MM-DD.zip`, one per day, each containing a single
  CSV. The Baltic / Øresund (Cluster 12) reproduction uses the window
  **2024-12-03 .. 2024-12-17** (15 files).
- Daily files are large (~2–4 GB uncompressed). The loader reads `.zip`, `.csv`,
  `.zst`, and `.gz` directly, so no manual extraction is needed, and accepts a
  whole directory of daily zips at once.
- An optional helper, `dma_validation/download_dma.py`, downloads a date range
  locally (run on a machine with internet access).

The DMA CSV schema differs from NOAA and is normalized by `dma_loader.py`:

| Field | Notes |
|-------|-------|
| Header | First column is `# Timestamp` (leading `#` comment marker), stripped on load |
| `Timestamp` | **Day-first** format `DD/MM/YYYY HH:MM:SS` (e.g. `03/12/2024` = 3 December) |
| `Latitude` / `Longitude` | `91` / `181` are AIS "not available" sentinels, dropped |
| `Ship type` | Textual category (`Cargo`, `Tanker`, ...), kept as a string |

A small **synthetic** fixture (`aisdk-2024-12-03_SAMPLE.zip`) is bundled so the
module runs without the full download; it is generated for code testing only and
is not real AIS data.

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

The `noaa_validation/` module uses these same defaults, defined in
`PipelineConfig` inside `anomaly_pipeline.py`.

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

### NOAA Reproduction (open data)

End-to-end recovery of the Gulf of Mexico (Cluster 10) event from NOAA data:

```bash
cd noaa_validation
pip install pandas numpy matplotlib zstandard

# Quick check on the bundled sample (~5 MB, runs in seconds):
python run_gulf_validation.py --input AIS_2024_12_31_SAMPLE.zip --day 2024-12-31 --outdir sample_results

# Full reproduction on the official NOAA daily file:
python run_gulf_validation.py --input AIS_2024_12_31.zip --day 2024-12-31 --outdir results

# Control day:
python run_gulf_validation.py --input ais-2025-01-01.csv.zst --day 2025-01-01 --outdir results_jan01
```

Outputs (per run): `gulf_flagged_vessels.csv`, `gulf_episodes.csv`,
`gulf_synchronization.csv`, and `gulf_validation.png`. The driver prints
lower/upper-bound vessel counts and a confirmation verdict. See
`noaa_validation/README.md` for full details and the headline result
(NOAA recovers **10 / 634** vessels vs. Spire's **13 / 1,196** for the same zone,
with the same linear, cross-shaped geometry).

### DMA Reproduction (open data)

Multi-day run over the Baltic / Øresund (Cluster 12) from DMA data:

```bash
cd dma_validation
pip install pandas numpy matplotlib zstandard

# Quick check on the bundled synthetic sample:
python run_baltic_validation.py --input aisdk-2024-12-03_SAMPLE.zip --outdir sample_results

# Full window: point at the folder of downloaded daily zips (read while zipped):
python run_baltic_validation.py --input /path/to/dma_files --outdir results

# Or a glob / explicit file list:
python run_baltic_validation.py --input "/path/to/dma_files/aisdk-2024-12-*.zip" --outdir results
```

The runner consumes multiple daily files at once, stitches multi-day vessel
tracks, and defaults to the Cluster 12 home box `(54.5, 56.5, 11.0, 14.0)`
(center `(55.4358, 12.7245)`); override with `--bbox LATMIN LATMAX LONMIN LONMAX`.
Because the Øresund is dense, the loader thins to one fix per vessel per minute
by default; control this with `--thin-seconds N` (`0` disables).

Outputs (per run): `baltic_flagged_vessels.csv`, `baltic_episodes.csv`,
`baltic_synchronization.csv`, `baltic_per_day.csv`, plus two figures —
`baltic_validation.png` (map + synchronization timeline + per-day bars) and
`baltic_jumps.png` (standalone jump map, styled like the Gulf figure). The
driver prints lower/upper-bound vessel counts and a per-day breakdown. See
`dma_validation/README.md` for full details.

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

As an additional open-data check, the `noaa_validation/` module independently
recovers the Gulf of Mexico event from NOAA's public feed. Comparing the event
day (Dec 31) against a control day (Jan 1) isolates the event (7 Dec-31-specific
vessels) from a small persistent baseline (3 faulty/default-MMSI transmitters
present on both days), consistent with the paper's classification of Cluster 10
as an isolated single-day event.

## Known Limitations

1. **Stealthy spoofing**: Physically consistent false trajectories may evade detection
2. **Self-spoofing ambiguity**: Cannot always distinguish self-spoofing from external interference when co-located
3. **Regional coverage gaps**: Detection sensitivity varies with AIS coverage density
4. **NOAA coverage**: The open-data reproduction is bounded by U.S. terrestrial-receiver range, so it provides a lower-bound view of any event whose spoofed positions extend beyond U.S. coastal waters

## Citation

If you use this code in your research, please cite:

```
[Citation information withheld for anonymous review]
```

## License

This code is provided for academic review purposes. See LICENSE file for terms.

## Contact

For questions regarding this submission, please contact the authors through the conference review system.
