# Data Format Specifications
# Maritime GPS Spoofing Detection Framework

This document describes the expected data formats for the spoofing detection pipeline.

## Input Data Formats

### AIS Route Data (Primary Input)

The pipeline expects AIS data in gzipped JSON format where each file contains
vessel routes keyed by MMSI.

**File format**: `*.json.gz`

**Schema**:
```json
{
  "MMSI_1": [
    ["timestamp_iso", latitude, longitude, speed_knots, course_degrees],
    ["timestamp_iso", latitude, longitude, speed_knots, course_degrees],
    ...
  ],
  "MMSI_2": [
    ...
  ]
}
```

**Field definitions**:
- `MMSI`: 9-digit Maritime Mobile Service Identity (string key)
- `timestamp_iso`: ISO 8601 timestamp, e.g., "2024-12-15T14:30:00Z"
- `latitude`: Decimal degrees, range [-90, 90]
- `longitude`: Decimal degrees, range [-180, 180]
- `speed_knots`: Speed over ground in knots (nullable)
- `course_degrees`: Course over ground in degrees [0, 360) (nullable)

**Example**:
```json
{
  "123456789": [
    ["2024-12-15T14:30:00Z", 45.5234, -122.6762, 12.5, 180.0],
    ["2024-12-15T14:35:00Z", 45.5200, -122.6762, 12.3, 181.2],
    ["2024-12-15T14:40:00Z", 45.5166, -122.6763, 12.4, 180.5]
  ],
  "987654321": [
    ["2024-12-15T14:32:00Z", 37.7749, -122.4194, 8.2, 45.0],
    ...
  ]
}
```

### Global Surface Water Raster (Optional)

For land detection, the pipeline uses the JRC Global Surface Water Occurrence dataset.

**Format**: GeoTIFF or VRT mosaic
**Resolution**: ~30m (0.0003 degrees)
**Values**: Water occurrence percentage (0-100)
**Source**: https://global-surface-water.appspot.com/

## Output Data Formats

### Stage 1 Results (Per-Vessel Detection)

**File format**: `*_results.json.gz`

**Schema**:
```json
[
  {
    "mmsi": "123456789",
    "point_count": 150,
    "violations": [
      {
        "index": 45,
        "timestamp": "2024-12-15T16:30:00",
        "latitude": 45.5234,
        "longitude": -122.6762,
        "reason": "deviation",
        "value": 7.523,
        "threshold": 5.0
      }
    ],
    "episodes": [
      {
        "start_idx": 40,
        "end_idx": 85,
        "start_timestamp": "2024-12-15T16:15:00",
        "end_timestamp": "2024-12-15T18:30:00",
        "duration_minutes": 135.0,
        "point_count": 46,
        "reasons": "deviation:32,speed:8"
      }
    ],
    "status": "flagged",
    "skip_reason": null
  }
]
```

**Status values**:
- `flagged`: Spoofing episodes detected
- `clean`: No violations detected
- `violations_only`: Violations detected but no episodes confirmed
- `skipped`: Track filtered (see `skip_reason`)
- `error`: Processing error

**Violation reasons**:
- `deviation`: Kalman residual exceeded threshold
- `speed`: Speed exceeded physical limit
- `land`: Position classified as on land

### Stage 2 Results (Cluster Analysis)

**File format**: `cluster_assignments.json`

**Schema**:
```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "vessel_count": 45,
      "vessels": ["123456789", "987654321", ...],
      "centroid_lat": 45.123,
      "centroid_lon": -122.456,
      "intersection_count": 127,
      "temporal_start": "2024-12-01T00:00:00",
      "temporal_end": "2024-12-31T23:59:59"
    }
  ],
  "self_spoofers": ["111111111", "222222222", ...],
  "metadata": {
    "epsilon_km": 50.0,
    "min_samples": 5,
    "total_vessels_processed": 2663
  }
}
```

### Temporal Classification Results

**File format**: `temporal_classifications.json`

**Schema**:
```json
{
  "0": {
    "cluster_id": 0,
    "pattern": "sustained",
    "confidence": 0.85,
    "metrics": {
      "duration_days": 14.5,
      "max_count": 45,
      "mean_count": 23.4,
      "peak_count": 3,
      "daily_cv": 0.08
    }
  }
}
```

**Pattern values**:
- `sustained`: Continuous activity over 7+ days
- `recurrent`: Periodic activity with regular intervals
- `intermittent`: Irregular sporadic bursts
- `isolated`: Single short-duration event

## Coordinate Reference System

All coordinates use WGS84 (EPSG:4326):
- Latitude: Degrees north of equator, positive north
- Longitude: Degrees east of prime meridian, positive east

## Timestamp Format

All timestamps use ISO 8601 format in UTC:
- Format: `YYYY-MM-DDTHH:MM:SSZ` or `YYYY-MM-DDTHH:MM:SS+00:00`
- Example: `2024-12-15T14:30:00Z`

## MMSI Format

Maritime Mobile Service Identity:
- 9 decimal digits
- Leading zeros preserved (string format)
- First digit indicates station type (2-7 for ships)
- Digits 2-4 indicate flag state (MID code)
