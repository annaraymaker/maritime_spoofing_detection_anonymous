"""
run_baltic_validation.py
========================

Reproduce the Baltic Sea near Copenhagen (Cluster 12) anomaly zone from the
Danish Maritime Authority (DMA) public AIS feed.

Unlike the Gulf of Mexico event (a single isolated day), Cluster 12 is an
*intermittent* zone: the paper's window for it is roughly 2024-12-03 .. 12-17.
This runner therefore takes MULTIPLE daily files, builds multi-day vessel
tracks, runs the per-vessel pipeline once over the whole window, and reports:

  * aggregate vessel counts (lower bound = FSM-confirmed; upper = any violation)
  * a PER-DAY breakdown of confirmed-vessel activity (to expose the
    intermittent bursts)
  * displacement geometry (the paper labels Cluster 12 "linear displacements")
  * a multi-day 1-minute synchronization timeline + figure

Usage
-----
    # point at the directory of downloaded DMA daily zips ...
    python run_baltic_validation.py --input /path/to/dma_files/ --outdir results

    # ... or list the days explicitly ...
    python run_baltic_validation.py \
        --input aisdk-2024-12-03.zip aisdk-2024-12-04.zip ... aisdk-2024-12-17.zip \
        --outdir results

    # ... or use a glob:
    python run_baltic_validation.py --input "aisdk-2024-12-*.zip" --outdir results
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from dma_loader import load_vessels_present_in_box
from anomaly_pipeline import PipelineConfig, analyze_vessel
from zone_analysis import classify_geometry, home_centroid

# Baltic / Oresund home box, anchored on Cluster 12 center (55.4358, 12.7245)
# near Copenhagen. Covers the Oresund strait and the southwestern Baltic
# approaches without spilling into the German Bight or the Gulf of Bothnia.
DEFAULT_BALTIC_BBOX = (54.5, 56.5, 11.0, 14.0)  # (lat_min, lat_max, lon_min, lon_max)


def run(inputs, outdir: str, bbox=DEFAULT_BALTIC_BBOX, cfg=None, thin_seconds: int = 60):
    cfg = cfg or PipelineConfig()
    os.makedirs(outdir, exist_ok=True)

    print(f"[1/4] Loading vessels present in Baltic box {bbox} across all input days ...")
    data = load_vessels_present_in_box(inputs, bbox, thin_seconds=thin_seconds)
    if data.empty:
        print("      No data loaded. Check input paths.")
        return
    n_vessels = data["mmsi"].nunique()
    span = (data["timestamp"].min(), data["timestamp"].max())
    print(f"      {len(data):,} reports from {n_vessels:,} home-region vessels.")
    print(f"      Time span: {span[0]} .. {span[1]}")

    print("[2/4] Running per-vessel anomaly pipeline (Kalman + violations + FSM) ...")
    vessel_rows, all_episodes = [], []
    episodes_by_vessel, flagged_points_by_vessel = {}, {}
    upper_bound_mmsi = set()

    for mmsi, track in data.groupby("mmsi"):
        flagged, episodes = analyze_vessel(track, cfg)
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

    vessels = pd.DataFrame(
        vessel_rows,
        columns=["mmsi", "n_episodes", "total_anom_min", "axis",
                 "linear_r2", "max_jump_km", "n_anom_points"],
    )
    if len(vessels):
        vessels = vessels.sort_values("max_jump_km", ascending=False)
    episodes_df = pd.concat(all_episodes, ignore_index=True) if all_episodes else pd.DataFrame()
    N = len(vessels)
    N_upper = len(upper_bound_mmsi)
    print(f"      {N} vessels with confirmed episodes (lower bound); "
          f"{N_upper} with any violation (upper bound).")

    print("[3/4] Per-day breakdown + multi-day synchronization ...")
    tl, peak, peak_time, active_min = _multiday_timeline(episodes_by_vessel)
    per_day = _per_day_breakdown(episodes_by_vessel)

    axis_counts = vessels["axis"].value_counts().to_dict() if N else {}
    has_ew = axis_counts.get("E-W", 0) + axis_counts.get("cross", 0)
    has_ns = axis_counts.get("N-S", 0) + axis_counts.get("cross", 0)
    linear_like = int((vessels["linear_r2"] > 0.95).sum()) if N else 0

    vessels.to_csv(os.path.join(outdir, "baltic_flagged_vessels.csv"), index=False)
    if not episodes_df.empty:
        episodes_df.to_csv(os.path.join(outdir, "baltic_episodes.csv"), index=False)
    tl.to_csv(os.path.join(outdir, "baltic_synchronization.csv"), index=False)
    per_day.to_csv(os.path.join(outdir, "baltic_per_day.csv"), index=False)

    print("[4/4] Rendering figure ...")
    _make_figure(flagged_points_by_vessel, tl, per_day, outdir, bbox)
    _make_jump_map(flagged_points_by_vessel, outdir, bbox)

    print("\n" + "=" * 64)
    print("BALTIC / CLUSTER 12 (DMA) REPRODUCTION — SUMMARY")
    print("=" * 64)
    print(f"Window analyzed             : {span[0]} .. {span[1]}")
    print(f"Home-region vessels scanned : {n_vessels:,}")
    print(f"Vessels w/ confirmed eps (N): {N}   [lower bound]")
    print(f"Vessels w/ any violation    : {N_upper}   [upper bound]")
    print(f"Total anomaly episodes      : {0 if episodes_df.empty else len(episodes_df)}")
    print(f"Displacement axis breakdown : {axis_counts}")
    print(f"Linear-geometry vessels(R2>.95): {linear_like} / {N}")
    print(f"Peak simultaneous vessels   : {peak} at {peak_time}")
    print(f"Active days (>=1 confirmed) : "
          f"{int((per_day['confirmed_vessels'] > 0).sum())} / {len(per_day)}")
    print("Per-day confirmed vessels:")
    for _, r in per_day.iterrows():
        bar = "#" * int(r["confirmed_vessels"])
        print(f"   {r['date']}  {int(r['confirmed_vessels']):3d}  {bar}")
    print("-" * 64)
    # Cluster 12 is "Likely" / linear / intermittent. A reasonable confirmation
    # is: linear geometry dominates AND activity recurs on multiple days.
    verdict = (N >= 5) and (linear_like >= max(1, N // 2)) and \
              (int((per_day["confirmed_vessels"] > 0).sum()) >= 3)
    print("VERDICT: " + (
        "Intermittent linear-displacement pattern CONSISTENT with Cluster 12."
        if verdict else
        "Pattern NOT clearly confirmed under current thresholds (inspect outputs)."
    ))
    print("=" * 64)
    return {"N": N, "N_upper": N_upper, "peak": int(peak),
            "axis_counts": axis_counts, "verdict": verdict}


def _multiday_timeline(episodes_by_vessel):
    """1-minute concurrency timeline spanning the whole multi-day window."""
    starts, ends = [], []
    for eps in episodes_by_vessel.values():
        for _, e in eps.iterrows():
            starts.append(pd.Timestamp(e["start"]))
            ends.append(pd.Timestamp(e["end"]))
    if not starts:
        idx = pd.date_range("2024-12-03", "2024-12-17 23:59", freq="min")
        tl = pd.DataFrame({"minute": idx, "active_vessels": 0})
        return tl, 0, idx[0], 0
    lo = min(starts).floor("h")
    hi = max(ends).ceil("h")
    idx = pd.date_range(lo, hi, freq="min")
    counter = pd.Series(0, index=idx, dtype=int)
    for eps in episodes_by_vessel.values():
        for _, e in eps.iterrows():
            counter.loc[pd.Timestamp(e["start"]).floor("min"):
                        pd.Timestamp(e["end"]).ceil("min")] += 1
    tl = counter.reset_index()
    tl.columns = ["minute", "active_vessels"]
    return tl, int(counter.max()), counter.idxmax(), int((counter > 0).sum())


def _per_day_breakdown(episodes_by_vessel):
    """Count distinct confirmed vessels active on each calendar day."""
    by_day: dict[str, set] = {}
    for mmsi, eps in episodes_by_vessel.items():
        for _, e in eps.iterrows():
            for d in pd.date_range(pd.Timestamp(e["start"]).normalize(),
                                   pd.Timestamp(e["end"]).normalize(), freq="D"):
                by_day.setdefault(d.strftime("%Y-%m-%d"), set()).add(mmsi)
    rows = [{"date": d, "confirmed_vessels": len(s)} for d, s in sorted(by_day.items())]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "confirmed_vessels"])


def _make_figure(flagged_points_by_vessel, tl, per_day, outdir, bbox):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.5))

    for mmsi, fp in flagged_points_by_vessel.items():
        clean = fp[~fp["violation"]]; anom = fp[fp["violation"]]
        hlat, hlon = home_centroid(fp)
        for la, lo in zip(anom["lat"], anom["lon"]):
            ax1.plot([hlon, lo], [hlat, la], c="#EE7733", lw=0.15, alpha=0.25)
        ax1.scatter(clean["lon"], clean["lat"], s=2, c="#4477AA", alpha=0.4)
        ax1.scatter(anom["lon"], anom["lat"], s=4, c="#EE7733", alpha=0.6)
    la0, la1, lo0, lo1 = bbox
    ax1.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0],
             c="#228833", lw=1.2, ls="--", label="Baltic home box")
    ax1.set_title("Cluster 12 flagged vessels (Baltic / Oresund)\n"
                  "blue = clean/true, orange = anomalous/spoofed")
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.fill_between(tl["minute"], tl["active_vessels"], step="mid",
                     color="#EE7733", alpha=0.75)
    ax2.set_title("Simultaneously anomalous vessels (1-min)")
    ax2.set_xlabel("Time (UTC)"); ax2.set_ylabel("Active flagged vessels")
    for lab in ax2.get_xticklabels():
        lab.set_rotation(30); lab.set_ha("right")

    if len(per_day):
        ax3.bar(per_day["date"], per_day["confirmed_vessels"], color="#4477AA")
    ax3.set_title("Confirmed-episode vessels per day")
    ax3.set_xlabel("Date"); ax3.set_ylabel("Distinct vessels")
    for lab in ax3.get_xticklabels():
        lab.set_rotation(60); lab.set_ha("right")

    fig.tight_layout()
    out = os.path.join(outdir, "baltic_validation.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"      figure -> {out}")


def _make_jump_map(flagged_points_by_vessel, outdir, bbox):
    """Standalone jump map, styled like the Gulf figure (single panel)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    for mmsi, fp in flagged_points_by_vessel.items():
        clean = fp[~fp["violation"]]; anom = fp[fp["violation"]]
        hlat, hlon = home_centroid(fp)
        for la, lo in zip(anom["lat"], anom["lon"]):
            ax.plot([hlon, lo], [hlat, la], c="#EE7733", lw=0.2, alpha=0.3)
        ax.scatter(clean["lon"], clean["lat"], s=3, c="#4477AA", alpha=0.4)
        ax.scatter(anom["lon"], anom["lat"], s=5, c="#EE7733", alpha=0.6)
    la0, la1, lo0, lo1 = bbox
    ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0],
            c="#228833", lw=1.3, ls="--", label="Cluster 12 home box")
    ax.set_title("Cluster 12 (Baltic / Oresund) spoofing jumps\n"
                 "blue = clean / true position, orange = anomalous / spoofed (lines = jumps)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(outdir, "baltic_jumps.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"      jump map -> {out}")


def main():
    ap = argparse.ArgumentParser(description="DMA Baltic / Cluster 12 reproduction.")
    ap.add_argument("--input", nargs="+", required=True,
                    help="DMA file(s), a glob, or a directory of aisdk-*.zip")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    ap.add_argument("--thin-seconds", type=int, default=60,
                    help="Keep >=1 fix per vessel per N seconds (memory control; "
                         "0 disables). Default 60.")
    args = ap.parse_args()
    bbox = tuple(args.bbox) if args.bbox else DEFAULT_BALTIC_BBOX
    run(args.input, args.outdir, bbox=bbox, thin_seconds=args.thin_seconds)


if __name__ == "__main__":
    main()
