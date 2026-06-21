#!/usr/bin/env python3
"""End-to-end spoofing demo with ground truth.

Pipeline:
  1. build a synthetic GPS+Galileo constellation and a known ground track;
  2. generate clean raw measurements, then spoof a chosen *subset* of satellites
     (self-consistent with a false position, with the AGC/C-N0 RF signature);
  3. write both as Android GnssLogger CSV logs (so the real-data path is used);
  4. stream the spoofed log through the real-time anti-spoofing engine;
  5. score detection + the position error of the naive vs mitigated solution
     against the ground truth, and write CSV/KML tracks.

Run:
    python apps/simulate_attack.py
    python apps/simulate_attack.py --spoof G02,G07,E04 --start 40 --end 100 --offset 800,300,0
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.coordinates import ecef_to_geodetic                      # noqa: E402
from src.core.output import write_csv, write_kml                       # noqa: E402
from src.io.gnsslogger import load_epochs                              # noqa: E402
from src.realtime.engine import RealtimeEngine                         # noqa: E402
from src.simulate.gnsslogger_writer import write_gnsslogger_csv        # noqa: E402
from src.simulate.scenario import (                                    # noqa: E402
    make_synthetic_constellation, make_trajectory, simulate_track,
)
from src.simulate.spoofer import SpoofConfig, apply_spoofing           # noqa: E402
from src.antispoof.detector import SpoofStatus                         # noqa: E402

M_PER_DEG_LAT = 111_132.0


def horiz_error(lat, lon, tlat, tlon):
    dlat = (lat - tlat) * M_PER_DEG_LAT
    dlon = (lon - tlon) * M_PER_DEG_LAT * math.cos(math.radians(tlat))
    return math.sqrt(dlat * dlat + dlon * dlon)


def main():
    ap = argparse.ArgumentParser(description="GNSS subset-spoofing demo with ground truth")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--spoof", default="G06,G10,G17,E04",
                    help="comma-separated PRNs to spoof (a subset of the visible satellites)")
    ap.add_argument("--start", type=int, default=50, help="attack start epoch")
    ap.add_argument("--end", type=int, default=120, help="attack end epoch (inclusive)")
    ap.add_argument("--offset", default="700,250,0", help="false-position ENU offset metres")
    ap.add_argument("--no-ramp", action="store_true")
    ap.add_argument("--noise", type=float, default=0.6, help="pseudorange noise std (m)")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    spoof_prns = [p.strip() for p in args.spoof.split(",") if p.strip()]
    ex, ey, ez = (float(v) for v in args.offset.split(","))
    isb_true = {"E": 12.0}

    print("=" * 70)
    print("GNSS-AntiSpoof — subset spoofing demo")
    print("=" * 70)
    nav, ref = make_synthetic_constellation()
    traj = make_trajectory(n_epochs=args.epochs, ref_time=ref)
    print(f"constellation: {sum(len(v) for v in nav['G'].values())} GPS + "
          f"{sum(len(v) for v in nav['E'].values())} Galileo")
    print(f"track: {args.epochs} epochs @1Hz, moving east; receiver ISB(E)={isb_true['E']} m")

    clean = simulate_track(traj, nav, isb_true=isb_true, noise_m=args.noise)
    cfg = SpoofConfig(
        target_prns=spoof_prns, start_epoch=args.start, end_epoch=args.end,
        false_offset_enu=(ex, ey, ez), ramp=not args.no_ramp,
    )
    spoofed, meta = apply_spoofing(clean, traj, nav, cfg, isb_true=isb_true)

    n_vis0 = clean[0].n_sats
    print(f"\nattack: spoofing {spoof_prns} ({len(spoof_prns)}/{n_vis0} sats) "
          f"epochs {args.start}-{args.end}, false offset E={ex} N={ey} U={ez} m, "
          f"ramp={'on' if not args.no_ramp else 'off'}")

    clean_csv = os.path.join(args.out, "scenario_clean.csv")
    spoof_csv = os.path.join(args.out, "scenario_spoofed.csv")
    write_gnsslogger_csv(clean, clean_csv, header_comment="clean scenario")
    write_gnsslogger_csv(spoofed, spoof_csv, header_comment="spoofed scenario")
    print(f"wrote GnssLogger logs: {clean_csv}, {spoof_csv}")

    # ---- run engine over the spoofed log via the real-data parser ----
    epochs = load_epochs(spoof_csv)
    engine = RealtimeEngine(nav, leap_seconds=18, gate_m=120.0,
                            physical_baseline_epochs=min(8, args.start - 2))
    outputs = engine.run(epochs)

    # ---- scoring ----
    naive_err_attack, mitig_err_attack = [], []
    naive_err_clean, mitig_err_clean = [], []
    tp = fp = fn = tn = 0          # epoch-level spoof detection
    id_tp = id_fp = id_fn = 0      # per-PRN identification during attack
    first_detect = None
    rows = []

    for i, o in enumerate(outputs):
        t = traj[i]
        attack_on = meta.active_by_epoch[i]
        truth_spoofed = set(meta.spoofed_prns_by_epoch[i])
        detected = o.status == SpoofStatus.SPOOFED

        if attack_on:
            if detected:
                tp += 1
                if first_detect is None:
                    first_detect = i - args.start
            else:
                fn += 1
            ident = set(o.spoofed_prns)
            id_tp += len(ident & truth_spoofed)
            id_fp += len(ident - truth_spoofed)
            id_fn += len(truth_spoofed - ident)
        else:
            if detected:
                fp += 1
            else:
                tn += 1

        if o.lat_deg is not None:
            me = horiz_error(o.lat_deg, o.lon_deg, t.lat_deg, t.lon_deg)
            (mitig_err_attack if attack_on else mitig_err_clean).append(me)
        if o.naive_lat_deg is not None:
            ne = horiz_error(o.naive_lat_deg, o.naive_lon_deg, t.lat_deg, t.lon_deg)
            (naive_err_attack if attack_on else naive_err_clean).append(ne)

        rows.append({
            "utc_time": o.utc_time, "latitude_deg": o.lat_deg, "longitude_deg": o.lon_deg,
            "altitude_m": o.alt_m, "speed_mps": o.speed_mps, "n_sats": o.n_sats,
            "pdop": o.pdop, "hdop": o.hdop,
        })

    def rms(v):
        return math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")

    print("\n" + "-" * 70)
    print("DETECTION (epoch-level, during attack):")
    n_attack = tp + fn
    print(f"  attack epochs: {n_attack}   detected: {tp}   missed: {fn}   "
          f"-> detection rate {100*tp/max(n_attack,1):.1f}%")
    print(f"  clean epochs:  {tn+fp}   false alarms: {fp}   "
          f"-> false-alarm rate {100*fp/max(tn+fp,1):.1f}%")
    if first_detect is not None:
        print(f"  detection latency: {first_detect} epoch(s) after attack onset")
    print(f"\nSPOOFED-SATELLITE IDENTIFICATION (per-PRN, during attack):")
    prec = id_tp / max(id_tp + id_fp, 1)
    rec = id_tp / max(id_tp + id_fn, 1)
    print(f"  correctly flagged: {id_tp}   false flags: {id_fp}   missed: {id_fn}")
    print(f"  precision {100*prec:.1f}%   recall {100*rec:.1f}%")

    print(f"\nPOSITION ERROR vs ground truth (horizontal RMS):")
    print(f"  during attack — naive (unprotected): {rms(naive_err_attack):8.2f} m")
    print(f"  during attack — mitigated (engine):  {rms(mitig_err_attack):8.2f} m")
    print(f"  clean epochs  — mitigated:           {rms(mitig_err_clean):8.2f} m")
    improvement = rms(naive_err_attack) / max(rms(mitig_err_attack), 1e-6)
    print(f"  -> mitigation reduces attack-time error {improvement:.1f}x")

    # ---- write tracks ----
    out_csv = os.path.join(args.out, "engine_track.csv")
    out_kml = os.path.join(args.out, "engine_track.kml")
    write_csv(out_csv, rows)
    write_kml(out_kml, rows, name="GNSS-AntiSpoof mitigated track")
    truth_rows = [{"latitude_deg": t.lat_deg, "longitude_deg": t.lon_deg, "altitude_m": t.alt_m}
                  for t in traj]
    write_kml(os.path.join(args.out, "truth_track.kml"), truth_rows, name="ground truth")
    print(f"\nwrote {out_csv}, {out_kml}, truth_track.kml")
    print("=" * 70)


if __name__ == "__main__":
    main()
