# NOAA Gulf of Mexico Reproduction

This module independently reproduces the **Gulf of Mexico (Cluster 10)**
spoofing event using **NOAA's public AIS feed**, separate from the Spire
dataset used in the main study. It exists to support the paper's claim that
the synchronized displacement pattern observed on December 31 is recoverable
from open data alone.

The code mirrors **Stage 1** of the methodology (per-vessel anomaly analysis:
Kalman prediction, violation checks, FSM episode grouping) and a single-region
version of **Stage 2** (geometry classification and temporal synchronization)
for one known zone, so reviewers can run it end-to-end on a single NOAA daily
file.

## Layout

| File | Role |
|------|------|
| `noaa_loader.py` | Schema-normalizing loader. Handles both NOAA layouts (classic `MMSI,BaseDateTime,LAT,LON,...` and the lowercase `mmsi,base_date_time,longitude,latitude,...` variant with swapped lon/lat) and transparently reads `.csv` / `.zip` / `.zst` / `.gz`. |
| `anomaly_pipeline.py` | Per-vessel Stage 1: cleaning, constant-velocity Kalman filter, the three violation checks (5 km deviation, 60 kn speed, optional on-land), and the start-by-quorum / end-by-consecutive FSM. All thresholds match the paper and live in `PipelineConfig`. |
| `zone_analysis.py` | Single-region Stage 2: displacement-geometry classification (E-W / N-S / cross / diagonal, with a linear-fit R²) and a 1-minute vessel-concurrency timeline. |
| `run_gulf_validation.py` | Driver. Filters to the Gulf home box, runs the pipeline, prints lower/upper-bound vessel counts, and writes result tables plus a figure. |

## Running

```bash
pip install pandas numpy matplotlib zstandard
python run_gulf_validation.py --input AIS_2024_12_31.csv --day 2024-12-31 --outdir results
```

### Quick start on the bundled sample (no download needed)

A ~5 MB sample (`AIS_2024_12_31_SAMPLE.zip`, 70 vessels) is bundled so the module
runs in seconds and still reproduces the event:

```bash
python run_gulf_validation.py --input AIS_2024_12_31_SAMPLE.zip --day 2024-12-31 --outdir sample_results
# -> N = 10 confirmed-episode vessels, cross-shaped pattern CONFIRMED
```

`sample_ais_variant_schema.csv` is a second sample in NOAA's lowercase /
swapped-coordinate layout, included to exercise the loader's schema handling.

### Full data

Download official daily files from
`https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/index.html`. The paper
uses **Dec 31, 2024** (event) and **Jan 1, 2025** (control).

`--input` accepts the file in any supported form (e.g. the NOAA `.zip` directly,
or a `.zst`). Override the region with `--bbox LATMIN LATMAX LONMIN LONMAX` and
the day with `--day`. The Gulf home box defaults to
`(26.0, 30.5, -97.5, -88.0)`, anchored on the Cluster 10 center
`(29.4044, -95.0533)`.

## What "validation" means here

Spoofing displaces a vessel's *reported* position far from its true location,
so the loader uses a **two-pass** strategy: pass 1 finds every MMSI that reports
inside the Gulf box (its home region); pass 2 keeps each such vessel's **full**
daily track, including the jumped-away points. Geometry and synchronization are
then measured per the paper.

We report two bounds, matching the paper's convention:

* **Lower bound** — vessels with an FSM-confirmed episode (≥70 % anomalous over
  a 30-minute window, ≥30 min long).
* **Upper bound** — vessels with at least one violating point on the day.

## Result on the provided NOAA files

| | Dec 31, 2024 (event) | Jan 1, 2025 (control) |
|---|---|---|
| Home-region vessels scanned | 3,514 | 3,385 |
| Confirmed-episode vessels (lower bound) | **10** | 5 |
| Any-violation vessels (upper bound) | 634 | 408 |
| Peak simultaneous flagged vessels | 6 | 4 |
| Displacement geometry | linear, cross-shaped (E-W + N-S), median fit R² ≈ 1.00 | weaker |

* The Spire-based Gulf cluster reported **13 (lower) / 1,196 (upper)** vessels.
  NOAA — an independent, US-coastal-only feed — recovers **10 / 634**, the same
  order of magnitude and the same **linear, cross-shaped** geometry radiating
  from the Texas–Louisiana coast.
* Comparing the two days isolates the event from baseline noise: **3** vessels
  are anomalous on *both* days (persistent faulty/default-MMSI transmitters),
  while **7** are Dec-31-specific — consistent with the paper's classification
  of Cluster 10 as an *isolated single-day* event.

### Caveats

* NOAA's feed comes from **U.S. terrestrial receivers**: it covers vessels in
  U.S. coastal waters **regardless of flag** (~16% of Gulf-box vessels are
  foreign-flagged), so it is not a U.S.-flag-only feed. Its counts run lower than
  the Spire (global satellite) figures because it cannot see spoofed endpoints
  that jump outside U.S. receiver range or the broader global fleet — a coverage
  difference, not a flag restriction.
* The upper bound is loose by construction (a single bad fix qualifies) and
  mixes in benign GPS glitches; the lower bound is the robust figure.
* A few flagged MMSIs are placeholder/default values
  (`200000000`, `300000000`, `360000000`, `982000000`) more consistent with
  faulty transmitters than external spoofing. The main study separates these
  via metadata-consistency checks (Section 3.4); this single-file reproduction
  reports them but flags the day-over-day comparison as the cleaner signal.
* The on-land check is optional and off by default (it needs the Global Surface
  Water mask, which is not bundled). In the Gulf the deviation and speed checks
  dominate, so it is not required to recover the event.

### Outputs written to `--outdir`

* `gulf_flagged_vessels.csv` — per-vessel geometry, episode count, max jump.
* `gulf_episodes.csv` — every confirmed episode with violation breakdown.
* `gulf_synchronization.csv` — the 1-minute concurrency timeline.
* `gulf_validation.png` — flagged-vessel map (with jump lines) + timeline.
