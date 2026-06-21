#!/usr/bin/env python3
"""Run the real-time anti-spoofing engine on a real recording.

Inputs (the *real-data path*):
  * an Android GnssLogger 'Raw' CSV  (--gnsslog), or
  * a RINEX 3/4 observation file     (--obs),
  plus broadcast navigation ephemeris (--nav, a RINEX nav / BRDC file).

Outputs (under --out):
  * <stem>_track.csv   — per-epoch trusted PVT + spoof status + diagnostics
  * <stem>_track.kml   — trusted (green) vs naive (red) tracks, spoof markers
  * <stem>_spoof.csv   — only the SUSPECT/SPOOFED epochs with reasons

Examples:
    python apps/run_realtime.py --gnsslog data/raw/gnss_log.csv --nav data/brdc/BRDC_169.rnx
    python apps/run_realtime.py --obs data/raw/rec.26o --nav data/brdc/BRDC_169.rnx
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io import gnsslogger, rinex_stream                              # noqa: E402
from src.realtime.engine import RealtimeEngine                           # noqa: E402
from src.realtime.reporting import (                                     # noqa: E402
    print_summary, write_engine_csv, write_engine_kml, write_spoof_events,
)


def main():
    ap = argparse.ArgumentParser(description="Real-time GNSS anti-spoofing on a recording")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gnsslog", help="Android GnssLogger 'Raw' CSV path")
    src.add_argument("--obs", help="RINEX 3/4 observation file path")
    ap.add_argument("--nav", required=True, help="RINEX navigation / BRDC file path")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--leap-seconds", type=int, default=18)
    ap.add_argument("--elev-mask-deg", type=float, default=10.0)
    ap.add_argument("--gate-m", type=float, default=120.0,
                    help="prior-gate distance for the majority-spoof fallback")
    ap.add_argument("--sigma-rho", type=float, default=5.0, help="nominal pseudorange sigma (m)")
    ap.add_argument("--baseline-epochs", type=int, default=8,
                    help="epochs assumed clean to learn the AGC/C-N0 baseline")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nav_data = rinex_stream.load_nav(args.nav)
    counts = {k: sum(len(v) for v in d.values()) for k, d in nav_data.items() if d}
    print(f"loaded nav ephemeris: {counts}")

    if args.gnsslog:
        epochs = gnsslogger.load_epochs(args.gnsslog, leap_seconds=args.leap_seconds)
        slots = {}
        stem = os.path.splitext(os.path.basename(args.gnsslog))[0]
        print(f"parsed GnssLogger CSV: {len(epochs)} epochs")
    else:
        epochs = rinex_stream.load_epochs(args.obs)
        slots = rinex_stream.read_glonass_slots(args.obs)
        stem = os.path.splitext(os.path.basename(args.obs))[0]
        print(f"parsed RINEX OBS: {len(epochs)} epochs")

    if not epochs:
        sys.exit("no epochs parsed — check the input file format")

    engine = RealtimeEngine(
        nav_data, glonass_slot_to_k=slots, leap_seconds=args.leap_seconds,
        elevation_mask_deg=args.elev_mask_deg, gate_m=args.gate_m,
        sigma_rho=args.sigma_rho, physical_baseline_epochs=args.baseline_epochs,
    )
    outputs = engine.run(epochs)

    csv_path = os.path.join(args.out, f"{stem}_track.csv")
    kml_path = os.path.join(args.out, f"{stem}_track.kml")
    spoof_path = os.path.join(args.out, f"{stem}_spoof.csv")
    write_engine_csv(csv_path, outputs)
    write_engine_kml(kml_path, outputs, name=f"{stem} (GNSS-AntiSpoof)")
    n_events = write_spoof_events(spoof_path, outputs)

    print()
    print_summary(outputs)
    print(f"\nwrote {csv_path}")
    print(f"wrote {kml_path}")
    print(f"wrote {spoof_path}  ({n_events} flagged epochs)")


if __name__ == "__main__":
    main()
