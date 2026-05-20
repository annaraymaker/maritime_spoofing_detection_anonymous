"""
zone_analysis.py
================

Stage-2-style aggregation for a single known region (here, the Gulf of
Mexico, Cluster 10). Where the paper's full Stage 2 clusters intersections
globally with DBSCAN, a single-region reproduction only needs to answer three
concrete questions about the vessels Stage 1 flagged:

  1. HOW MANY distinct vessels show confirmed anomaly episodes?  -> N
  2. GEOMETRY: are the displacements the "cross-shaped" mix of E-W (horizontal,
     constant-latitude) and N-S (vertical, constant-longitude) linear jumps the
     paper reports for the Gulf?
  3. SYNCHRONIZATION: do the episodes overlap in time (a coherent field) rather
     than scattering randomly across the day?

These three together constitute the paper's claim: "the same synchronized
displacement pattern across [N] vessels on December 31."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly_pipeline import haversine_km


def home_centroid(flagged: pd.DataFrame):
    """
    Estimate a vessel's true ("home") position as the centroid of its
    non-violating points. Falls back to the median of all points if every
    point is flagged.
    """
    clean = flagged[~flagged["violation"]]
    if len(clean) >= 3:
        return clean["lat"].median(), clean["lon"].median()
    return flagged["lat"].median(), flagged["lon"].median()


def classify_geometry(flagged: pd.DataFrame, axis_tol_deg: float = 0.5):
    """
    Classify a vessel's anomalous displacement geometry relative to its home
    position.

    Returns dict with:
        axis        : 'E-W' | 'N-S' | 'cross' | 'diagonal' | 'none'
        linear_r2   : R^2 of a straight-line fit to anomalous points (0..1)
        max_jump_km : largest displacement of an anomalous point from home
        n_anom      : number of anomalous points considered
    """
    anom = flagged[flagged["violation"]]
    if len(anom) < 3:
        return {"axis": "none", "linear_r2": np.nan, "max_jump_km": 0.0, "n_anom": len(anom)}

    hlat, hlon = home_centroid(flagged)
    dlat = (anom["lat"] - hlat).to_numpy()
    dlon = (anom["lon"] - hlon).to_numpy()

    # How much of the displacement is along latitude vs longitude.
    lat_spread = np.percentile(np.abs(dlat), 90)
    lon_spread = np.percentile(np.abs(dlon), 90)

    # Points that move mainly E-W (lat ~ constant) vs mainly N-S (lon ~ constant).
    ew = np.mean(np.abs(dlat) < axis_tol_deg)   # fraction holding latitude
    ns = np.mean(np.abs(dlon) < axis_tol_deg)   # fraction holding longitude

    if ew > 0.5 and ns > 0.5:
        axis = "cross"
    elif ew > 0.5 and lon_spread > axis_tol_deg:
        axis = "E-W"
    elif ns > 0.5 and lat_spread > axis_tol_deg:
        axis = "N-S"
    else:
        axis = "diagonal"

    # Straight-line fit quality on the anomalous points (linear-displacement test).
    if len(anom) >= 3 and (anom["lat"].std() + anom["lon"].std()) > 1e-6:
        x = anom["lon"].to_numpy()
        y = anom["lat"].to_numpy()
        # Total-least-squares via PCA: r2 = explained variance of 1st component.
        X = np.column_stack([x - x.mean(), y - y.mean()])
        if np.allclose(X, 0):
            r2 = np.nan
        else:
            cov = np.cov(X.T)
            evals = np.linalg.eigvalsh(cov)
            r2 = float(evals.max() / evals.sum()) if evals.sum() > 0 else np.nan
    else:
        r2 = np.nan

    max_jump = float(
        haversine_km(hlat, hlon, anom["lat"].to_numpy(), anom["lon"].to_numpy()).max()
    )
    return {"axis": axis, "linear_r2": r2, "max_jump_km": max_jump, "n_anom": len(anom)}


def synchronization_timeline(episodes_by_vessel: dict[int, pd.DataFrame], day: str):
    """
    Build a 1-minute timeline of how many distinct vessels are simultaneously
    inside a confirmed anomaly episode.

    Returns (timeline_df, peak_count, peak_time, active_minutes).
    """
    idx = pd.date_range(f"{day} 00:00", f"{day} 23:59", freq="min")
    counter = pd.Series(0, index=idx, dtype=int)
    for _mmsi, eps in episodes_by_vessel.items():
        for _, e in eps.iterrows():
            s = pd.Timestamp(e["start"]).floor("min")
            en = pd.Timestamp(e["end"]).ceil("min")
            counter.loc[s:en] += 1
    peak_count = int(counter.max())
    peak_time = counter.idxmax()
    active_minutes = int((counter > 0).sum())
    tl = counter.reset_index()
    tl.columns = ["minute", "active_vessels"]
    return tl, peak_count, peak_time, active_minutes
