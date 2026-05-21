"""
anomaly_pipeline.py
===================

Per-vessel anomaly identification, mirroring Stage 1 of the paper
(Section 3.3, Figure 3). Given one vessel's timestamp-ordered track, it:

  1. cleans the track (dedup, sort, drop stationary transmitters)
  2. runs a constant-velocity linear Kalman filter for position prediction
  3. applies three violation checks: deviation > 5 km, speed > 60 kn, on-land
  4. groups point-level violations into temporally coherent "anomaly episodes"
     with a finite-state machine (start-by-quorum / end-by-consecutive)

Parameters are the paper's defaults and live in `PipelineConfig` so the repo
and the reproduction run use exactly the same numbers.

The on-land check requires an external water mask (Global Surface Water) that
is not bundled here; it is therefore optional and OFF by default. In the Gulf
of Mexico reproduction it is unnecessary: the synchronized jumps are caught by
the deviation and speed checks, which dominate. Enable `check_land` and pass a
`land_fn(lat, lon) -> bool` if you have a mask available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

R_EARTH_KM = 6371.0


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    # Cleaning
    stationary_radius_km: float = 0.5      # drop transmitters that never leave this radius

    # Kalman
    process_noise: float = 1e-5            # Q (paper: 1e-5)
    measurement_noise: float = 1e-4        # R (paper: 1e-4)
    kalman_reset_gap_min: float = 7.0      # re-init after gaps > 7 min

    # Violation thresholds
    deviation_km: float = 5.0              # Kalman residual > 5 km  -> deviation violation
    speed_max_kn: float = 60.0             # instantaneous speed > 60 kn -> speed violation
    check_land: bool = False               # optional on-land check (needs land_fn)

    # FSM (minute-binned)
    rolling_window_min: int = 30           # RW
    quorum_ratio: float = 0.70             # MV = ceil(0.70 * N_RW)
    clean_gap_end_min: int = 120           # CGE: consecutive clean minutes to close episode
    min_episode_min: int = 30              # discard episodes shorter than this


# --------------------------------------------------------------------------- #
# Geodesy                                                                      #
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km). Vectorized; handles +/-180 wrap implicitly."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


# --------------------------------------------------------------------------- #
# Stage 1.2  Cleaning                                                          #
# --------------------------------------------------------------------------- #
def clean_track(track: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Sort, dedup identical (timestamp, position), drop stationary transmitters."""
    t = track.sort_values("timestamp").copy()
    t = t.drop_duplicates(subset=["timestamp", "lat", "lon"])
    if len(t) < 2:
        return t
    # Stationary check: full spatial extent within stationary_radius_km => moored/base station.
    span = haversine_km(t["lat"].min(), t["lon"].min(), t["lat"].max(), t["lon"].max())
    if span < cfg.stationary_radius_km:
        return t.iloc[0:0]  # empty -> excluded
    return t.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 1.3  Kalman prediction + 1.4 violation checks                          #
# --------------------------------------------------------------------------- #
def _kalman_predict_residuals(t: pd.DataFrame, cfg: PipelineConfig) -> np.ndarray:
    """
    Constant-velocity linear Kalman filter over (lat, lon, dlat, dlon).
    Returns the geodesic residual (km) between each prediction and the
    observed AIS position. The first point of every (re)initialized segment
    has residual 0 (nothing to predict from yet).

    Filter is re-initialized whenever the gap to the previous point exceeds
    cfg.kalman_reset_gap_min, because state error accumulates with sparse
    updates (paper: 7-minute reset, >92% of gaps are shorter).
    """
    lat = t["lat"].to_numpy()
    lon = t["lon"].to_numpy()
    ts = t["timestamp"].to_numpy()
    n = len(t)
    resid = np.zeros(n)

    Q = cfg.process_noise
    Rm = cfg.measurement_noise

    x = None  # state vector [lat, lon, dlat, dlon]
    P = None  # covariance
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    Qm = np.eye(4) * Q
    Rmat = np.eye(2) * Rm

    prev_t = None
    for i in range(n):
        if i == 0:
            x = np.array([lat[i], lon[i], 0.0, 0.0])
            P = np.eye(4)
            prev_t = ts[i]
            continue

        dt_h = (ts[i] - prev_t) / np.timedelta64(1, "h")
        if dt_h <= 0 or dt_h * 60.0 > cfg.kalman_reset_gap_min:
            # Re-initialize: gap too large (or non-monotonic) to trust prediction.
            x = np.array([lat[i], lon[i], 0.0, 0.0])
            P = np.eye(4)
            prev_t = ts[i]
            resid[i] = 0.0
            continue

        # Predict
        F = np.array(
            [[1, 0, dt_h, 0], [0, 1, 0, dt_h], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        x_pred = F @ x
        P_pred = F @ P @ F.T + Qm

        # Residual between predicted position and the new observation.
        pred_lat, pred_lon = x_pred[0], x_pred[1]
        resid[i] = haversine_km(pred_lat, pred_lon, lat[i], lon[i])

        # Update (measurement)
        z = np.array([lat[i], lon[i]])
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + Rmat
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred
        prev_t = ts[i]

    return resid


def flag_violations(
    t: pd.DataFrame,
    cfg: PipelineConfig,
    land_fn: Optional[Callable[[float, float], bool]] = None,
) -> pd.DataFrame:
    """
    Add per-point violation columns:
        kalman_resid_km, dev_violation, speed_violation, land_violation, violation
    `violation` is the OR of the enabled checks.
    """
    t = t.copy()
    if len(t) < 2:
        t["kalman_resid_km"] = 0.0
        t["dev_violation"] = False
        t["speed_violation"] = False
        t["land_violation"] = False
        t["violation"] = False
        return t

    # Deviation via Kalman residual.
    t["kalman_resid_km"] = _kalman_predict_residuals(t, cfg)
    t["dev_violation"] = t["kalman_resid_km"] > cfg.deviation_km

    # Instantaneous implied speed between consecutive points.
    plat = t["lat"].shift()
    plon = t["lon"].shift()
    pt = t["timestamp"].shift()
    seg_km = haversine_km(plat, plon, t["lat"], t["lon"])
    dt_h = (t["timestamp"] - pt).dt.total_seconds() / 3600.0
    implied_kn = seg_km / 1.852 / dt_h.replace(0, np.nan)
    t["implied_kn"] = implied_kn
    t["speed_violation"] = implied_kn > cfg.speed_max_kn

    # Optional land check.
    if cfg.check_land and land_fn is not None:
        t["land_violation"] = [
            bool(land_fn(la, lo)) for la, lo in zip(t["lat"], t["lon"])
        ]
    else:
        t["land_violation"] = False

    t["violation"] = (
        t["dev_violation"].fillna(False)
        | t["speed_violation"].fillna(False)
        | t["land_violation"].fillna(False)
    )
    return t


# --------------------------------------------------------------------------- #
# Stage 1.5  FSM grouping into episodes                                        #
# --------------------------------------------------------------------------- #
def group_episodes(t: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Aggregate point-level violations into anomaly episodes using the paper's
    start-by-quorum / end-by-consecutive FSM on minute-binned data.

    Returns a DataFrame of episodes with columns:
        start, end, duration_min, n_points, n_violations,
        dev_frac, speed_frac, land_frac
    """
    if len(t) < 2 or not t["violation"].any():
        return pd.DataFrame(
            columns=[
                "start", "end", "duration_min", "n_points",
                "n_violations", "dev_frac", "speed_frac", "land_frac",
            ]
        )

    t = t.copy()
    t["minute"] = t["timestamp"].dt.floor("min")

    # Minute is anomalous if ANY point in it violates.
    per_min = (
        t.groupby("minute")
        .agg(
            anom=("violation", "any"),
            n=("violation", "size"),
            dev=("dev_violation", "any"),
            spd=("speed_violation", "any"),
            land=("land_violation", "any"),
        )
        .reset_index()
    )

    # Build a continuous minute index so the rolling window sees real time gaps.
    full_idx = pd.date_range(per_min["minute"].min(), per_min["minute"].max(), freq="min")
    per_min = per_min.set_index("minute").reindex(full_idx)
    per_min["anom"] = per_min["anom"].fillna(False)
    minutes = per_min.index.to_numpy()
    anom = per_min["anom"].to_numpy().astype(bool)

    RW = cfg.rolling_window_min
    CGE = cfg.clean_gap_end_min

    episodes = []
    state = "IDLE"
    ep_start = None
    clean_run = 0

    for i in range(len(minutes)):
        # Rolling window ending at i (last RW minutes).
        lo = max(0, i - RW + 1)
        win = anom[lo : i + 1]
        n_win = len(win)
        a_win = int(win.sum())
        mv = math.ceil(cfg.quorum_ratio * n_win)
        quorum = (n_win == RW) and (a_win >= mv)

        if state == "IDLE":
            if anom[i]:
                state = "CANDIDATE"
                ep_start = minutes[i]
        elif state == "CANDIDATE":
            if quorum:
                state = "ACTIVE"
                clean_run = 0
            elif a_win == 0:
                state = "IDLE"
                ep_start = None
        elif state == "ACTIVE":
            if anom[i]:
                clean_run = 0
            else:
                clean_run += 1
                if clean_run >= CGE:
                    ep_end = minutes[i - clean_run + 1]  # last anomalous minute
                    episodes.append((ep_start, ep_end))
                    state = "IDLE"
                    ep_start = None
                    clean_run = 0

    if state == "ACTIVE" and ep_start is not None:
        episodes.append((ep_start, minutes[-1]))

    # Build episode records, discarding sub-threshold durations.
    recs = []
    for s, e in episodes:
        dur = (pd.Timestamp(e) - pd.Timestamp(s)).total_seconds() / 60.0
        if dur < cfg.min_episode_min:
            continue
        pts = t[(t["timestamp"] >= s) & (t["timestamp"] <= e)]
        nv = int(pts["violation"].sum())
        recs.append(
            {
                "start": pd.Timestamp(s),
                "end": pd.Timestamp(e),
                "duration_min": dur,
                "n_points": len(pts),
                "n_violations": nv,
                "dev_frac": float(pts["dev_violation"].mean()) if len(pts) else 0.0,
                "speed_frac": float(pts["speed_violation"].mean()) if len(pts) else 0.0,
                "land_frac": float(pts["land_violation"].mean()) if len(pts) else 0.0,
            }
        )
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Driver: run all of Stage 1 for one vessel                                    #
# --------------------------------------------------------------------------- #
def analyze_vessel(
    track: pd.DataFrame,
    cfg: PipelineConfig = None,
    land_fn=None,
):
    """
    Full Stage-1 pipeline for a single vessel.

    Returns (flagged_points_df, episodes_df). `flagged_points_df` is None if
    the vessel was dropped during cleaning (e.g. stationary base station).
    """
    cfg = cfg or PipelineConfig()
    cleaned = clean_track(track, cfg)
    if len(cleaned) < 2:
        return None, pd.DataFrame()
    flagged = flag_violations(cleaned, cfg, land_fn=land_fn)
    episodes = group_episodes(flagged, cfg)
    return flagged, episodes
