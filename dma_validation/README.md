# Baltic / Cluster 12 Reproduction (Danish Maritime Authority data)

Independent reproduction of the **Baltic Sea near Copenhagen (Cluster 12)**
anomaly zone using the **Danish Maritime Authority (DMA)** public AIS feed
(http://aisdata.ais.dk/), separate from the Spire dataset used in the main
study.

It reuses the same dataset-agnostic Stage-1 (`anomaly_pipeline.py`) and Stage-2
geometry/synchronization code (`zone_analysis.py`) as the NOAA module; only the
loader and the regional, multi-day runner differ.

Cluster 12 differs from the Gulf of Mexico event in two ways that shape this
module:

* It is **intermittent**, not a single isolated day. The paper's window is
  roughly **2024-12-03 .. 2024-12-17**, so the runner consumes *many* daily
  files, builds multi-day vessel tracks, and reports a per-day breakdown plus a
  multi-day synchronization timeline to expose the bursts.
* Its dominant geometry is **linear displacements** (paper Table 3), so the
  runner reports how many flagged vessels have a high straight-line fit
  (R² > 0.95) in addition to the E-W / N-S / cross axis breakdown.

## Layout

| File | Role |
|------|------|
| `dma_loader.py` | DMA schema loader. Maps the DMA header (`Timestamp, Type of mobile, MMSI, Latitude, Longitude, ... Ship type, ...`) to the canonical schema, parses **day-first** timestamps (`03/12/2024` = 3 Dec), drops the AIS "not available" sentinels (lat 91 / lon 181), and reads `.zip/.csv/.zst/.gz`. Supports multiple daily files, globs, or a directory. |
| `anomaly_pipeline.py` | Stage 1 — identical to the NOAA module (Kalman + 5 km deviation / 60 kn speed / optional on-land checks + start-by-quorum FSM). |
| `zone_analysis.py` | Geometry classification + concurrency timeline — identical to the NOAA module. |
| `run_baltic_validation.py` | Multi-day driver for Cluster 12. |
| `download_dma.py` | Optional helper to fetch a date range of DMA files **locally**. |
| `aisdk-2024-12-03_SAMPLE.zip` | **Synthetic** DMA-format fixture (see warning below). |

## Getting the data

The daily files are **not** included (they are large and must come from DMA).

1. Browse the listing: http://aisdata.ais.dk/?prefix=
2. Download `aisdk-2024-12-03.zip` through `aisdk-2024-12-17.zip` (15 files).
   Each is a single CSV, roughly 2-4 GB uncompressed, so budget ~30-50 GB.
   The optional helper automates the range:

   ```bash
   pip install requests
   python download_dma.py --start 2024-12-03 --end 2024-12-17 --outdir dma_files
   ```

   If the candidate URLs don't resolve, copy the real object URL from the
   listing page and pass `--base-url`.

> **Sandbox note.** This module was developed in an environment with no access
> to aisdata.ais.dk and no access to a local Downloads folder, so the real files
> could not be fetched here. Run `download_dma.py` (or download manually) on a
> machine with internet access, then point the runner at the files.

## DMA schema handled

```
Timestamp, Type of mobile, MMSI, Latitude, Longitude, Navigational status, ROT,
SOG, COG, Heading, IMO, Callsign, Name, Ship type, Cargo type, Width, Length,
Type of position fixing device, Draught, Destination, ETA, Data source type,
A, B, C, D
```

* The header's first column is literally `# Timestamp` (a leading comment
  marker); the loader strips the `#` so it maps correctly.
* Timestamps are **day-first** (`DD/MM/YYYY HH:MM:SS`) — handled.
* `Ship type` is textual (`Cargo`, `Tanker`, ...), kept as a string.
* Positions of `Latitude = 91` / `Longitude = 181` are AIS "not available"
  sentinels and are dropped.

## Running

```bash
pip install pandas numpy matplotlib zstandard

# Simplest: point at the folder of downloaded daily zips (read while zipped).
python run_baltic_validation.py --input ~/Downloads/temp --outdir results
```

The loader globs `aisdk-*.*` in a directory, reads each `.zip` in place (no
unzipping needed), and stitches multi-day tracks. You can also pass a glob
(`--input "~/Downloads/temp/aisdk-2024-12-*.zip"`) or list files explicitly.

**Memory.** The Øresund is extremely dense. To stay laptop-friendly the loader
thins each vessel to one fix per minute (`--thin-seconds 60`, the FSM bins by
minute anyway) and downcasts dtypes. A 15-day window typically holds well under
a GB. Lower it further (`--thin-seconds 120`) or tighten `--bbox` if memory is
tight; set `--thin-seconds 0` to disable thinning.

Region defaults to the Cluster 12 home box `(54.5, 56.5, 11.0, 14.0)`, anchored
on the cluster center `(55.4358, 12.7245)` near Copenhagen / the Øresund.
Override with `--bbox LATMIN LATMAX LONMIN LONMAX`.

### Smoke test on the bundled synthetic fixture

```bash
python run_baltic_validation.py --input aisdk-2024-12-03_SAMPLE.zip --outdir smoke_results
```

> **The bundled `aisdk-2024-12-03_SAMPLE.zip` is SYNTHETIC**, generated only to
> exercise the DMA loader and the full pipeline (schema parsing, day-first
> timestamps, sentinel filtering, FSM confirmation, geometry, per-day breakdown,
> figure). **It is not real AIS data and is not evidence of spoofing.** It
> contains a handful of normal tracks plus three planted teleport patterns so
> the confirmed-episode path is reachable. Delete it before using the module on
> real data if you prefer.

## Outputs (per run, in `--outdir`)

* `baltic_flagged_vessels.csv` — per-vessel geometry, episode count, max jump.
* `baltic_episodes.csv` — every confirmed episode with violation breakdown.
* `baltic_synchronization.csv` — multi-day 1-minute concurrency timeline.
* `baltic_per_day.csv` — distinct confirmed vessels active on each calendar day.
* `baltic_validation.png` — three panels: flagged-vessel map (with jump lines),
  multi-day synchronization timeline, and per-day confirmed-vessel bars.
* `baltic_jumps.png` — standalone single-panel jump map, styled like the Gulf
  figure, for direct visual comparison.

The driver prints lower/upper-bound vessel counts, the per-day breakdown as a
small text bar chart, and a verdict. The verdict is intentionally conservative
for an *intermittent* zone: it expects linear geometry to dominate and activity
to recur across multiple days.

## Caveats specific to the DMA feed

* **Coverage.** DMA primarily covers Danish and adjacent Baltic / North Sea
  waters via terrestrial and coastal receivers. Like the NOAA reproduction, it
  gives a conservative, regionally bounded view — spoofed endpoints that jump
  outside DMA receiver range are truncated.
* **Dense traffic.** The Øresund is one of the busiest straits in the world.
  Expect a larger raw home-region vessel count than the Gulf; the 5 km / 60 kn
  thresholds and the FSM persistence requirement are what keep false positives
  down (see the main study's port validation, Section 3.6).
* **MMSI reuse / faulty transmitters.** As in the Gulf, some flagged MMSIs will
  be reuse or hardware faults rather than external spoofing; compare the event
  window against a quiet control day (e.g. late November) to separate the event
  from baseline, exactly as Dec-31-vs-Jan-1 did for the Gulf.
* **On-land check** is off by default (needs the Global Surface Water mask); the
  deviation and speed checks dominate for linear-displacement zones.
