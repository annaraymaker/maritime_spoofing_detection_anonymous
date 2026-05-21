"""
download_dma.py
===============

Convenience downloader for Danish Maritime Authority daily AIS files, to be run
**locally on your own machine** (it needs internet access to aisdata.ais.dk).

The DMA daily objects are named `aisdk-YYYY-MM-DD.zip`. This script downloads a
date range. Because the exact object URL on the bucket can vary, it tries a few
candidate patterns and reports which worked; if none do, open the listing in a
browser (http://aisdata.ais.dk/?prefix=) and copy the real object URL, then set
--base-url accordingly.

Examples
--------
    # download the Cluster 12 window (Dec 3-17, 2024) into ./dma_files/
    python download_dma.py --start 2024-12-03 --end 2024-12-17 --outdir dma_files

    # if you discover the real base URL from the listing page:
    python download_dma.py --start 2024-12-03 --end 2024-12-17 \
        --base-url http://aisdata.ais.dk/ --outdir dma_files
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

try:
    import requests
except ImportError:
    sys.exit("This helper needs `requests`: pip install requests")

# Candidate URL patterns observed for DMA AIS objects. {d} = YYYY-MM-DD.
CANDIDATE_PATTERNS = [
    "http://aisdata.ais.dk/aisdk-{d}.zip",
    "http://aisdata.ais.dk/aisdk/{d}/aisdk-{d}.zip",
    "https://web.ais.dk/aisdata/aisdk-{d}.zip",
]


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def try_download(d: str, outdir: str, base_url: str | None) -> bool:
    patterns = [base_url + "aisdk-{d}.zip"] if base_url else CANDIDATE_PATTERNS
    dest = os.path.join(outdir, f"aisdk-{d}.zip")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  {d}: already present, skipping")
        return True
    for pat in patterns:
        url = pat.format(d=d)
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    continue
                total = int(r.headers.get("content-length", 0))
                print(f"  {d}: downloading {url}  ({total/1e6:.0f} MB)")
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                return True
        except requests.RequestException:
            continue
    print(f"  {d}: FAILED (no candidate URL worked) — check the listing page")
    return False


def main():
    ap = argparse.ArgumentParser(description="Download DMA daily AIS files.")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--outdir", default="dma_files")
    ap.add_argument("--base-url", default=None,
                    help="Override base URL, e.g. http://aisdata.ais.dk/ (must end in /)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    s = date.fromisoformat(args.start)
    e = date.fromisoformat(args.end)
    print(f"Downloading aisdk {s} .. {e} into {args.outdir}/")
    ok = sum(try_download(d.isoformat(), args.outdir, args.base_url) for d in daterange(s, e))
    print(f"Done: {ok} file(s) available in {args.outdir}/")


if __name__ == "__main__":
    main()
