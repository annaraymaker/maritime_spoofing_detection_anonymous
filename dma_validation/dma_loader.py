"""
dma_loader.py
=============

Schema-normalizing loader for Danish Maritime Authority (DMA) public AIS files
from http://aisdata.ais.dk/ (historically web.ais.dk/aisdata/).

The DMA distributes one file per day named `aisdk-YYYY-MM-DD.zip`, each
containing a single CSV `aisdk-YYYY-MM-DD.csv` with this header:

    Timestamp, Type of mobile, MMSI, Latitude, Longitude, Navigational status,
    ROT, SOG, COG, Heading, IMO, Callsign, Name, Ship type, Cargo type, Width,
    Length, Type of position fixing device, Draught, Destination, ETA,
    Data source type, A, B, C, D

Key differences from the NOAA feed that this loader handles:
  * Timestamp is DAY-FIRST: "03/12/2024 00:00:00" means 3 December, not 12 March.
  * Column is "Ship type" (text categories like "Cargo", "Tanker"), not a numeric
    code; we keep it as a string under `vessel_type`.
  * Sentinel/null positions: DMA encodes missing fixes as Latitude=91.0 and/or
    Longitude=181.0 (the AIS "not available" values). These are dropped.

Output is the SAME canonical schema as the NOAA loader, so the shared
anomaly_pipeline / zone_analysis modules consume it unchanged:

    mmsi int64 | timestamp datetime64 | lat float | lon float
    sog float  | cog float            | vessel_type (string, optional)

Reads .csv / .zip / .zst / .gz transparently, and supports loading several
daily files at once (the Cluster 12 event spans 2024-12-03 .. 2024-12-17).
"""

from __future__ import annotations

import glob
import io
import os
import zipfile
from typing import Iterable, Optional

import pandas as pd

# DMA header -> canonical name. Matching is case-insensitive and whitespace
# tolerant so minor header variations across years still resolve.
_DMA_ALIASES = {
    "timestamp": "timestamp",
    "mmsi": "mmsi",
    "latitude": "lat",
    "longitude": "lon",
    "sog": "sog",
    "cog": "cog",
    "ship type": "vessel_type",
}

_CANONICAL = ["mmsi", "timestamp", "lat", "lon", "sog", "cog", "vessel_type"]

# AIS "not available" sentinels used by the DMA feed.
_LAT_SENTINEL = 91.0
_LON_SENTINEL = 181.0


def _open_text(path: str) -> io.TextIOBase:
    """Text handle for csv/zip/zst/gz, decompressing on the fly."""
    lower = path.lower()
    if lower.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV inside zip: {path}")
        return io.TextIOWrapper(zf.open(names[0]), encoding="utf-8", errors="replace")
    if lower.endswith(".zst"):
        import zstandard as zstd

        fh = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    if lower.endswith(".gz"):
        import gzip

        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DMA columns to canonical schema and coerce dtypes (day-first time)."""
    rename = {}
    for c in df.columns:
        # The DMA header's first column is "# Timestamp" (a leading comment
        # marker), so strip any leading '#' and surrounding whitespace before
        # matching, then collapse internal whitespace.
        key = " ".join(c.strip().lstrip("#").strip().lower().split())
        if key in _DMA_ALIASES:
            rename[c] = _DMA_ALIASES[key]
    df = df.rename(columns=rename)

    keep = [c for c in _CANONICAL if c in df.columns]
    df = df[keep].copy()

    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce").astype("Int64")
    # DMA timestamps are day-first ("03/12/2024 00:00:00").
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", dayfirst=True
    )
    for col in ("lat", "lon", "sog", "cog"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "vessel_type" in df.columns:
        df["vessel_type"] = df["vessel_type"].astype("string")
    return df


def _basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot enter the motion pipeline (incl. DMA sentinels)."""
    df = df.dropna(subset=["mmsi", "timestamp", "lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    # DMA 'not available' sentinels and null island.
    df = df[~((df["lat"] >= _LAT_SENTINEL) | (df["lon"] >= _LON_SENTINEL))]
    df = df[~((df["lat"].abs() < 1e-6) & (df["lon"].abs() < 1e-6))]
    return df


def _iter_paths(inputs: Iterable[str]) -> list[str]:
    """Expand a mix of files, globs, and directories into a sorted file list."""
    paths: list[str] = []
    for item in inputs:
        if os.path.isdir(item):
            paths += glob.glob(os.path.join(item, "aisdk-*.*"))
        elif any(ch in item for ch in "*?[]"):
            paths += glob.glob(item)
        else:
            paths.append(item)
    # De-dup and sort so multi-day tracks come out in chronological order.
    return sorted(set(paths))


def load_vessels_present_in_box(
    inputs,
    bbox: tuple[float, float, float, float],
    chunksize: int = 1_000_000,
    thin_seconds: int = 60,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Two-pass, multi-day loader for trajectory analysis of a region.

    `inputs` may be a single path, a list of daily-file paths, a glob, or a
    directory of `aisdk-*.zip` files. Across ALL provided days:

      pass 1: find every MMSI that reports at least one position inside `bbox`.
      pass 2: return the FULL track (all days) for each such MMSI.

    Scalability
    -----------
    The Oresund is extremely dense, so a 15-day full-resolution load can be
    enormous. Two levers keep it laptop-friendly:

      * `thin_seconds` (default 60): keep at most one fix per vessel per time
        bin of this many seconds. The FSM bins by minute anyway, so 60 s loses
        nothing material while cutting row count by 1-2 orders of magnitude.
        Set to 0 to disable thinning.
      * dtype downcast: lat/lon/sog/cog -> float32, mmsi -> int32.

    Returns canonical-schema DataFrame sorted by (mmsi, timestamp).
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    paths = _iter_paths(inputs)
    if not paths:
        raise FileNotFoundError(f"No input files matched: {inputs}")

    lat_min, lat_max, lon_min, lon_max = bbox

    # Pass 1: home-region MMSIs across all days (low memory: just a set).
    home_mmsi: set[int] = set()
    for n, p in enumerate(paths, 1):
        if verbose:
            print(f"      pass 1 [{n}/{len(paths)}] {os.path.basename(p)} ...", flush=True)
        reader = pd.read_csv(_open_text(p), chunksize=chunksize, dtype=str)
        for chunk in reader:
            chunk = _basic_filter(_normalize(chunk))
            inbox = chunk[
                chunk["lat"].between(lat_min, lat_max)
                & chunk["lon"].between(lon_min, lon_max)
            ]
            home_mmsi.update(int(m) for m in inbox["mmsi"].dropna().unique())
    if verbose:
        print(f"      pass 1 done: {len(home_mmsi):,} home-region vessels.", flush=True)

    # Pass 2: full tracks for those MMSIs, thinned to >=1 fix / thin_seconds.
    parts: list[pd.DataFrame] = []
    for n, p in enumerate(paths, 1):
        if verbose:
            print(f"      pass 2 [{n}/{len(paths)}] {os.path.basename(p)} ...", flush=True)
        reader = pd.read_csv(_open_text(p), chunksize=chunksize, dtype=str)
        for chunk in reader:
            chunk = _basic_filter(_normalize(chunk))
            chunk = chunk[chunk["mmsi"].astype("float").isin(home_mmsi)]
            if not len(chunk):
                continue
            if thin_seconds and thin_seconds > 0:
                chunk = chunk.assign(
                    _tbin=chunk["timestamp"].dt.floor(f"{thin_seconds}s")
                ).drop_duplicates(subset=["mmsi", "_tbin"], keep="first")
            parts.append(chunk)

    if not parts:
        return pd.DataFrame(columns=_CANONICAL)
    out = pd.concat(parts, ignore_index=True)
    if thin_seconds and thin_seconds > 0 and "_tbin" in out.columns:
        out = out.drop_duplicates(subset=["mmsi", "_tbin"], keep="first").drop(columns=["_tbin"])
    out = out.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    # Downcast to shrink the in-memory footprint.
    out["mmsi"] = out["mmsi"].astype("int64").astype("int32")
    for col in ("lat", "lon", "sog", "cog"):
        if col in out.columns:
            out[col] = out[col].astype("float32")
    if verbose:
        mb = out.memory_usage(deep=True).sum() / 1e6
        print(f"      pass 2 done: {len(out):,} rows held ({mb:.0f} MB).", flush=True)
    return out


def load_ais(inputs, bbox=None, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Plain loader (optionally box-filtered) across one or more DMA files."""
    if isinstance(inputs, str):
        inputs = [inputs]
    paths = _iter_paths(inputs)
    parts: list[pd.DataFrame] = []
    for p in paths:
        reader = pd.read_csv(_open_text(p), chunksize=chunksize, dtype=str)
        for chunk in reader:
            chunk = _basic_filter(_normalize(chunk))
            if bbox is not None:
                la0, la1, lo0, lo1 = bbox
                chunk = chunk[chunk["lat"].between(la0, la1) & chunk["lon"].between(lo0, lo1)]
            if len(chunk):
                parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=_CANONICAL)
    out = pd.concat(parts, ignore_index=True).sort_values(["mmsi", "timestamp"])
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import sys

    df = load_ais(sys.argv[1:], chunksize=500_000)
    print(f"{len(df):,} rows, {df['mmsi'].nunique():,} vessels")
    print(df.head())
    print(df.dtypes)
