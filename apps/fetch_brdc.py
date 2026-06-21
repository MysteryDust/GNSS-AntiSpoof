#!/usr/bin/env python3
"""Download mixed RINEX-3 broadcast navigation (BRDC) files for given dates.

Use this to get the ephemeris that matches a real Android/RINEX recording, then
feed it to apps/run_realtime.py via --nav.

Usage:
    python apps/fetch_brdc.py 2026-06-18
    python apps/fetch_brdc.py 2026-06-17 2026-06-18
"""

import gzip
import os
import shutil
import ssl
import sys
import urllib.request
from datetime import datetime

BKG_TEMPLATE = (
    "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/"
    "BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "brdc")


def fetch_one(date_str: str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    doy = (dt - datetime(dt.year, 1, 1)).days + 1
    url = BKG_TEMPLATE.format(year=dt.year, doy=doy)
    os.makedirs(OUT_DIR, exist_ok=True)
    gz_path = os.path.join(OUT_DIR, f"BRDC_{dt.year}_{doy:03d}.rnx.gz")
    out_path = os.path.join(OUT_DIR, f"BRDC_{dt.year}_{doy:03d}.rnx")
    print(f"[fetch] {url}")
    try:
        resp = urllib.request.urlopen(url, timeout=90)
    except urllib.error.URLError as exc:
        # Some Python installs (notably Windows) lack a CA bundle and fail TLS
        # verification on this public server. Retry once without verification.
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
        print("        (TLS verify failed; retrying without certificate verification)")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        resp = urllib.request.urlopen(url, timeout=90, context=ctx)
    with resp, open(gz_path, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    with gzip.open(gz_path, "rb") as gf, open(out_path, "wb") as fh:
        shutil.copyfileobj(gf, fh)
    os.remove(gz_path)
    print(f"        -> {out_path}")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: fetch_brdc.py YYYY-MM-DD [YYYY-MM-DD ...]")
    for d in sys.argv[1:]:
        try:
            fetch_one(d)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed for {d}: {exc}")


if __name__ == "__main__":
    main()
