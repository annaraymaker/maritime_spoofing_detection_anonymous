"""
noaa_loader.py
==============

Schema-normalizing loader for NOAA / MarineCadastre public AIS daily files.

NOAA distributes daily AIS dumps in a few different layouts. This loader
accepts all of the variants we have encountered and emits a single canonical
schema so the rest of the pipeline never has to branch on format:

    canonical columns:
        mmsi          int64
        timestamp     datetime64[ns]   (UTC)
        lat           float64          (degrees, -90..90)
        lon           float64          (degrees, -180..180)
        sog           float64          (speed over ground, knots)
        cog           float64          (course over ground, degrees) -- optional
        vessel_type   Int64            -- optional, may be NaN

Known input variants
---------------------
1. Classic marinecadastre layout (e.g. AIS_2024_12_31.csv inside a .zip):
       MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,IMO,CallSign,
       VesselType,Status,Length,Width,Draft,Cargo,TransceiverClass
   - ISO timestamp "2024-12-31T00:00:08"
   - LAT before LON

2. Re-exported lowercase layout (e.g. ais-2025-01-01_csv.zst):
       mmsi,base_date_time,longitude,latitude,sog,cog,heading,vessel_name,
       imo,call_sign,vessel_type,status,length,width,draft,cargo,transceiver
   - Space timestamp "2025-01-01 00:00:00"
   - longitude BEFORE latitude (column order swapped)

The loader is transparent to compression: .zip, .zst, .gz, or plain .csv all
work, so the same file can be committed to the repo in whatever form is most
convenient.
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import Iterator, Optional

import pandas as pd


# Map every header spelling we have seen onto the canonical name.
_COLUMN_ALIASES = {
    "mmsi": "mmsi",
    "basedatetime": "timestamp",
    "base_date_time": "timestamp",
    "lat": "lat",
    "latitude": "lat",
    "lon": "lon",
    "longitude": "lon",
    "sog": "sog",
    "cog": "cog",
    "vesseltype": "vessel_type",
    "vessel_type": "vessel_type",
}

_CANONICAL = ["mmsi", "timestamp", "lat", "lon", "sog", "cog", "vessel_type"]


def _open_text(path: str) -> io.TextIOBase:
    """Return a text file handle for csv/zip/zst/gz, decompressing on the fly."""
    lower = path.lower()
    if lower.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        # NOAA daily zips contain exactly one CSV.
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV inside zip: {path}")
        return io.TextIOWrapper(zf.open(names[0]), encoding="utf-8")
    if lower.endswith(".zst"):
        import zstandard as zstd  # lazy import; only needed for .zst inputs

        fh = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8")
    if lower.endswith(".gz"):
        import gzip

        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical schema and coerce dtypes."""
    rename = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in _COLUMN_ALIASES:
            rename[c] = _COLUMN_ALIASES[key]
    df = df.rename(columns=rename)

    # Keep only canonical columns that are present.
    keep = [c for c in _CANONICAL if c in df.columns]
    df = df[keep].copy()

    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce").astype("Int64")
    # pandas parses both "2024-12-31T00:00:08" and "2025-01-01 00:00:00".
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    for col in ("lat", "lon", "sog", "cog"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "vessel_type" in df.columns:
        df["vessel_type"] = pd.to_numeric(df["vessel_type"], errors="coerce").astype("Int64")

    return df


def _basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot enter the motion pipeline."""
    df = df.dropna(subset=["mmsi", "timestamp", "lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    # 0,0 ("null island") is a classic bad fix, not a real position.
    df = df[~((df["lat"].abs() < 1e-6) & (df["lon"].abs() < 1e-6))]
    return df


def load_ais(
    path: str,
    bbox: Optional[tuple[float, float, float, float]] = None,
    chunksize: int = 1_000_000,
    usecols_only_canonical: bool = True,
) -> pd.DataFrame:
    """
    Load a NOAA AIS daily file into the canonical schema.

    Parameters
    ----------
    path : str
        Path to a .csv / .zip / .zst / .gz NOAA AIS file.
    bbox : (lat_min, lat_max, lon_min, lon_max), optional
        If given, keep only reports whose position falls inside the box.
        Filtering happens per-chunk so huge daily files never fully
        materialize in memory.
    chunksize : int
        Rows per read chunk.
    usecols_only_canonical : bool
        If True, only read columns we recognize (faster, lower memory).

    Returns
    -------
    pandas.DataFrame with canonical columns, sorted by (mmsi, timestamp).
    """
    handle = _open_text(path)
    try:
        # Peek the header to decide which columns to read.
        header = pd.read_csv(io.StringIO(handle.readline()), nrows=0)
        wanted = None
        if usecols_only_canonical:
            wanted = [c for c in header.columns if c.strip().lower() in _COLUMN_ALIASES]
        # Re-open: simplest robust path is to reopen the stream.
    finally:
        handle.close()

    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        _open_text(path),
        usecols=wanted,
        chunksize=chunksize,
        dtype=str,  # parse types ourselves in _normalize for consistency
    )
    for chunk in reader:
        chunk = _normalize(chunk)
        chunk = _basic_filter(chunk)
        if bbox is not None:
            lat_min, lat_max, lon_min, lon_max = bbox
            chunk = chunk[
                chunk["lat"].between(lat_min, lat_max)
                & chunk["lon"].between(lon_min, lon_max)
            ]
        if len(chunk):
            parts.append(chunk)

    if not parts:
        return pd.DataFrame(columns=_CANONICAL)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    return out


def load_vessels_present_in_box(
    path: str,
    bbox: tuple[float, float, float, float],
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """
    Two-pass loader for trajectory analysis of a region.

    Spoofing displaces a vessel's *reported* position far from its true
    location, so simply box-filtering positions would discard the spoofed
    endpoints we care about. Instead:

      pass 1: find every MMSI that reports at least one position inside `bbox`
              (its "home" region).
      pass 2: return the FULL daily track for each such MMSI, including any
              reports that jumped far outside the box.

    This is the correct way to study a regional spoofing zone: anchor on
    vessels that belong to the region, then keep their entire trajectory.
    """
    lat_min, lat_max, lon_min, lon_max = bbox

    # Pass 1: collect home-region MMSIs.
    home_mmsi: set[int] = set()
    reader = pd.read_csv(_open_text(path), chunksize=chunksize, dtype=str)
    for chunk in reader:
        chunk = _normalize(chunk)
        chunk = _basic_filter(chunk)
        inbox = chunk[
            chunk["lat"].between(lat_min, lat_max)
            & chunk["lon"].between(lon_min, lon_max)
        ]
        home_mmsi.update(int(m) for m in inbox["mmsi"].dropna().unique())

    # Pass 2: pull full tracks for those MMSIs.
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(_open_text(path), chunksize=chunksize, dtype=str)
    for chunk in reader:
        chunk = _normalize(chunk)
        chunk = _basic_filter(chunk)
        chunk = chunk[chunk["mmsi"].astype("float").isin(home_mmsi)]
        if len(chunk):
            parts.append(chunk)

    if not parts:
        return pd.DataFrame(columns=_CANONICAL)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    import sys

    p = sys.argv[1]
    df = load_ais(p, chunksize=500_000)
    print(f"{p}: {len(df):,} rows, {df['mmsi'].nunique():,} vessels")
    print(df.head())
    print(df.dtypes)
