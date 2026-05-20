"""
run_gulf_validation.py
======================

Reproduce the Gulf of Mexico (Cluster 10) spoofing event from the paper using
NOAA's public AIS feed, independently of the Spire dataset.

Usage
-----
    python run_gulf_validation.py \
        --input /path/to/AIS_2024_12_31.csv \
        --day 2024-12-31 \
        --outdir results/

Accepts .csv / .zip / .zst / .gz. The Gulf bounding box and all thresholds
default to the paper's values; override on the command line if needed.

Outputs
-------
    <outdir>/gulf_flagged_vessels.csv   per-vessel results (geometry, episodes)
    <outdir>/gulf_episodes.csv          every confirmed anomaly episode
    <outdir>/gulf_synchronization.csv   1-minute concurrency timeline
    <outdir>/gulf_validation.png        map + timeline figure
    prints the headline count N and a confirmation verdict
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from noaa_loader import load_vessels_present_in_box
from anomaly_pipeline import PipelineConfig, analyze_vessel
from zone_analysis import classify_geometry, synchronization_timeline

# Gulf of Mexico home box, anchored on Cluster 10 center (29.4044, -95.0533).
# Generous enough to capture the Texas/Louisiana shelf traffic that the event
# affected, tight enough to exclude unrelated Atlantic / Caribbean vessels.
DEFAULT_GULF_BBOX = (26.0, 30.5, -97.5, -88.0)  # (lat_min, lat_max, lon_min, lon_max)


def run(input_path: str, day: str, outdir: str, bbox=DEFAULT_GULF_BBOX, cfg=None):
    cfg = cfg or PipelineConfig()
    os.makedirs(outdir, exist_ok=True)

    print(f"[1/4] Loading vessels present in Gulf box {bbox} from {input_path} ...")
    data = load_vessels_present_in_box(input_path, bbox)
    n_vessels = data["mmsi"].nunique()
    print(f"      {len(data):,} reports from {n_vessels:,} home-region vessels.")

    print("[2/4] Running per-vessel anomaly pipeline (Kalman + violations + FSM) ...")
    vessel_rows = []
    all_episodes = []
    episodes_by_vessel = {}
    flagged_points_by_vessel = {}
    upper_bound_mmsi = set()  # any vessel with >=1 violating point (paper's upper bound)

    for mmsi, track in data.groupby("mmsi"):
        flagged, episodes = analyze_vessel(track, cfg)
        # Upper bound: vessel had at least one point violating any constraint,
        # regardless of whether the FSM confirmed a sustained episode.
        if flagged is not None and bool(flagged["violation"].any()):
            upper_bound_mmsi.add(int(mmsi))
        if flagged is None or episodes.empty:
            continue
        episodes = episodes.copy()
        episodes["mmsi"] = int(mmsi)
        all_episodes.append(episodes)
        episodes_by_vessel[int(mmsi)] = episodes
        flagged_points_by_vessel[int(mmsi)] = flagged

        geom = classify_geometry(flagged)
        vessel_rows.append(
            {
                "mmsi": int(mmsi),
                "n_episodes": len(episodes),
                "total_anom_min": float(episodes["duration_min"].sum()),
                "axis": geom["axis"],
                "linear_r2": geom["linear_r2"],
                "max_jump_km": geom["max_jump_km"],
                "n_anom_points": geom["n_anom"],
            }
        )

    vessels = pd.DataFrame(vessel_rows).sort_values("max_jump_km", ascending=False)
    episodes_df = (
        pd.concat(all_episodes, ignore_index=True) if all_episodes else pd.DataFrame()
    )

    N = len(vessels)
    N_upper = len(upper_bound_mmsi)
    print(f"      {N} vessels have confirmed anomaly episodes "
          f"(lower bound); {N_upper} vessels show >=1 violation (upper bound).")

    print("[3/4] Characterizing geometry and temporal synchronization ...")
    tl, peak, peak_time, active_min = synchronization_timeline(episodes_by_vessel, day)
    axis_counts = vessels["axis"].value_counts().to_dict() if N else {}
    # "Cross-shaped" presence = both horizontal (E-W) and vertical (N-S) axes
    # appear across the flagged fleet (a vessel may be one or the other; the
    # zone-level cross is the union).
    has_ew = axis_counts.get("E-W", 0) + axis_counts.get("cross", 0)
    has_ns = axis_counts.get("N-S", 0) + axis_counts.get("cross", 0)
    cross_present = has_ew > 0 and has_ns > 0

    # Save tables.
    vessels.to_csv(os.path.join(outdir, "gulf_flagged_vessels.csv"), index=False)
    if not episodes_df.empty:
        episodes_df.to_csv(os.path.join(outdir, "gulf_episodes.csv"), index=False)
    tl.to_csv(os.path.join(outdir, "gulf_synchronization.csv"), index=False)

    print("[4/4] Rendering figure ...")
    _make_figure(flagged_points_by_vessel, tl, outdir, day, bbox)

    # ------------------------------------------------------------------ #
    # Verdict
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 64)
    print("GULF OF MEXICO NOAA REPRODUCTION — SUMMARY")
    print("=" * 64)
    print(f"Day analyzed                : {day}")
    print(f"Home-region vessels scanned : {n_vessels:,}")
    print(f"Vessels w/ confirmed eps (N): {N}   [lower bound]")
    print(f"Vessels w/ any violation    : {N_upper}   [upper bound]")
    print(f"Total anomaly episodes      : {0 if episodes_df.empty else len(episodes_df)}")
    print(f"Displacement axis breakdown : {axis_counts}")
    print(f"  horizontal (E-W) vessels  : {has_ew}")
    print(f"  vertical   (N-S) vessels  : {has_ns}")
    print(f"Cross-shaped pattern present: {cross_present}")
    print(f"Peak simultaneous vessels   : {peak} at {peak_time}")
    print(f"Minutes with >=1 active eps : {active_min}")
    median_jump = float(vessels["max_jump_km"].median()) if N else 0.0
    print(f"Median max-jump per vessel  : {median_jump:.0f} km")
    verdict = (
        cross_present
        and N >= 10
        and peak >= 5
    )
    print("-" * 64)
    print(
        "VERDICT: "
        + (
            "Synchronized cross-shaped displacement pattern CONFIRMED "
            "in NOAA data."
            if verdict
            else "Pattern NOT clearly confirmed under current thresholds."
        )
    )
    print("=" * 64)
    return {
        "N": N,
        "N_upper": N_upper,
        "cross_present": cross_present,
        "peak": peak,
        "peak_time": str(peak_time),
        "axis_counts": axis_counts,
        "verdict": verdict,
    }


def _make_figure(flagged_points_by_vessel, tl, outdir, day, bbox):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: flagged-vessel points + jump lines from home to spoofed points,
    # which makes the horizontal/vertical "cross" emanating from the coast clear.
    from zone_analysis import home_centroid

    for mmsi, fp in flagged_points_by_vessel.items():
        clean = fp[~fp["violation"]]
        anom = fp[fp["violation"]]
        hlat, hlon = home_centroid(fp)
        # Draw faint lines from the home position to each anomalous point.
        for la, lo in zip(anom["lat"], anom["lon"]):
            ax1.plot([hlon, lo], [hlat, la], c="#EE7733", lw=0.15, alpha=0.25)
        ax1.scatter(clean["lon"], clean["lat"], s=2, c="#4477AA", alpha=0.4)
        ax1.scatter(anom["lon"], anom["lat"], s=4, c="#EE7733", alpha=0.6)
    # Draw home box.
    lat_min, lat_max, lon_min, lon_max = bbox
    ax1.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        c="#228833", lw=1.2, ls="--", label="Gulf home box",
    )
    ax1.set_title(f"Gulf of Mexico flagged vessels — {day}\n"
                  "blue = clean / true, orange = anomalous / spoofed (lines = jumps)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.legend(loc="upper right", fontsize=8)

    # Right: synchronization timeline.
    ax2.fill_between(tl["minute"], tl["active_vessels"], step="mid",
                     color="#EE7733", alpha=0.7)
    ax2.set_title("Simultaneously anomalous vessels (1-min resolution)")
    ax2.set_xlabel("Time (UTC)")
    ax2.set_ylabel("Active flagged vessels")
    for label in ax2.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    fig.tight_layout()
    out = os.path.join(outdir, "gulf_validation.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"      figure -> {out}")


def main():
    ap = argparse.ArgumentParser(description="NOAA Gulf of Mexico spoofing reproduction.")
    ap.add_argument("--input", required=True, help="NOAA AIS file (.csv/.zip/.zst/.gz)")
    ap.add_argument("--day", required=True, help="YYYY-MM-DD of the file")
    ap.add_argument("--outdir", default="results", help="output directory")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    args = ap.parse_args()
    bbox = tuple(args.bbox) if args.bbox else DEFAULT_GULF_BBOX
    run(args.input, args.day, args.outdir, bbox=bbox)


if __name__ == "__main__":
    main()
