#!/usr/bin/env python3
"""Generate evaluation figures (PNG) for the report / presentation.

Two modes:
  attack  — run a synthetic subset-spoofing attack and plot the true vs naive vs
            mitigated tracks and the horizontal error over time.
  real    — run the engine on a real GnssLogger/RINEX recording and plot the
            standalone track against the device's NMEA reference.

Requires matplotlib (`pip install -r requirements-plot.txt`). The core pipeline
itself needs no third-party packages — this is only for the figures.

Examples:
    python apps/evaluate.py attack --out docs/figures/attack.png
    python apps/evaluate.py real --gnsslog rec.txt --nav data/brdc/BRDC_2026_080.rnx \
        --nmea rec.nmea --out docs/figures/real_track.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from src.antispoof.detector import SpoofStatus                        # noqa: E402
from src.io import gnsslogger, rinex_stream                           # noqa: E402
from src.realtime.engine import RealtimeEngine                        # noqa: E402
from src.simulate.scenario import (                                   # noqa: E402
    make_synthetic_constellation, make_trajectory, simulate_track,
)
from src.simulate.spoofer import SpoofConfig, apply_spoofing          # noqa: E402

RED, GREEN, GRAY = "#E24B4A", "#1D9E75", "#888780"


def _enu(lat, lon, lat0, lon0):
    if lat is None:
        return None, None
    e = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
    n = (lat - lat0) * 111132.0
    return e, n


def attack_figure(out: str, start=50, end=120, offset=(700, 250, 0)):
    nav, ref = make_synthetic_constellation()
    traj = make_trajectory(n_epochs=150, ref_time=ref)
    isb = {"E": 12.0}
    clean = simulate_track(traj, nav, isb_true=isb, noise_m=0.6)
    cfg = SpoofConfig(target_prns=["G06", "G10", "G17", "E04"], start_epoch=start,
                      end_epoch=end, false_offset_enu=offset, ramp=False)
    spoofed, meta = apply_spoofing(clean, traj, nav, cfg, isb_true=isb)
    outs = RealtimeEngine(nav, gate_m=120.0, physical_baseline_epochs=8).run(spoofed)

    lat0, lon0 = traj[0].lat_deg, traj[0].lon_deg
    te, tn, ne, nn, me, mn, idx = [], [], [], [], [], [], []
    naive_err, mitig_err = [], []
    for i, o in enumerate(outs):
        t = traj[i]
        e, n = _enu(t.lat_deg, t.lon_deg, lat0, lon0); te.append(e); tn.append(n)
        e2, n2 = _enu(o.naive_lat_deg, o.naive_lon_deg, lat0, lon0); ne.append(e2); nn.append(n2)
        e3, n3 = _enu(o.lat_deg, o.lon_deg, lat0, lon0); me.append(e3); mn.append(n3)
        idx.append(i)
        naive_err.append(math.hypot(e2 - e, n2 - n) if e2 is not None else None)
        mitig_err.append(math.hypot(e3 - e, n3 - n) if e3 is not None else None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(te, tn, "--", color=GRAY, lw=1.5, label="true route")
    ax1.plot(ne, nn, "-", color=RED, lw=2, label="unprotected (dragged by spoof)")
    ax1.plot(me, mn, "-", color=GREEN, lw=2.5, label="protected (mitigated)")
    ax1.set_aspect("equal"); ax1.set_xlabel("east (m)"); ax1.set_ylabel("north (m)")
    ax1.set_title("Subset spoofing: 4 of 13 satellites spoofed (epochs %d–%d)" % (start, end))
    ax1.legend(loc="upper left", fontsize=9); ax1.grid(alpha=0.2)

    ax2.axvspan(start, end, color=RED, alpha=0.10, label="attack window")
    ax2.plot(idx, naive_err, color=RED, lw=1.6, label="unprotected error")
    ax2.plot(idx, mitig_err, color=GREEN, lw=2, label="protected error")
    ax2.set_xlabel("epoch (s)"); ax2.set_ylabel("horizontal error (m)")
    ax2.legend(loc="upper right", fontsize=9); ax2.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def _parse_gga(path):
    fixes = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        if "GGA" not in line:
            continue
        p = line.split(",")
        hit = [k for k, t in enumerate(p) if t.endswith("GGA")]
        if not hit:
            continue
        i = hit[0]
        try:
            hh = p[i + 1].split(".")[0]
            latdm = float(p[i + 2]); lath = p[i + 3]; londm = float(p[i + 4]); lonh = p[i + 5]
        except (ValueError, IndexError):
            continue
        lat = int(latdm / 100) + (latdm - 100 * int(latdm / 100)) / 60.0
        lon = int(londm / 100) + (londm - 100 * int(londm / 100)) / 60.0
        if lath == "S":
            lat = -lat
        if lonh == "W":
            lon = -lon
        fixes[hh] = (lat, lon)
    return fixes


def real_figure(out: str, gnsslog=None, obs=None, nav=None, nmea=None):
    nav_data = rinex_stream.load_nav(nav)
    if gnsslog:
        eps = gnsslogger.load_epochs(gnsslog); slots = {}
    else:
        eps = rinex_stream.load_epochs(obs); slots = rinex_stream.read_glonass_slots(obs)
    outs = RealtimeEngine(nav_data, glonass_slot_to_k=slots, gate_m=120.0,
                          physical_baseline_epochs=10).run(eps)
    ref = _parse_gga(nmea) if nmea else {}

    lat0 = lon0 = None
    re_, rn_, ee_, en_, errs, eidx = [], [], [], [], [], []
    for o in outs:
        if o.lat_deg is None:
            continue
        if lat0 is None:
            lat0, lon0 = o.lat_deg, o.lon_deg
        e, n = _enu(o.lat_deg, o.lon_deg, lat0, lon0); ee_.append(e); en_.append(n)
        hh = o.utc_time.strftime("%H%M%S")
        if hh in ref:
            re, rn = _enu(ref[hh][0], ref[hh][1], lat0, lon0); re_.append(re); rn_.append(rn)
            errs.append(math.hypot(e - re, n - rn)); eidx.append(len(errs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    if re_:
        ax1.plot(re_, rn_, "-", color=GRAY, lw=2.5, label="device NMEA")
    ax1.plot(ee_, en_, "-", color=GREEN, lw=1.2, label="standalone (this engine)")
    ax1.set_aspect("equal"); ax1.set_xlabel("east (m)"); ax1.set_ylabel("north (m)")
    ax1.set_title("Standalone track vs device"); ax1.legend(fontsize=9); ax1.grid(alpha=0.2)
    if errs:
        errs_sorted = sorted(errs)
        med = errs_sorted[len(errs_sorted) // 2]
        ax2.hist(errs, bins=40, color=GREEN, alpha=0.8)
        ax2.axvline(med, color=RED, lw=1.5, label="median %.1f m" % med)
        ax2.set_xlabel("horizontal error vs NMEA (m)"); ax2.set_ylabel("epochs")
        ax2.set_title("Error distribution"); ax2.legend(fontsize=9); ax2.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Generate evaluation figures")
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("attack")
    a.add_argument("--out", default="docs/figures/attack.png")
    r = sub.add_parser("real")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--gnsslog")
    src.add_argument("--obs")
    r.add_argument("--nav", required=True)
    r.add_argument("--nmea")
    r.add_argument("--out", default="docs/figures/real_track.png")
    args = ap.parse_args()
    if args.mode == "attack":
        attack_figure(args.out)
    else:
        real_figure(args.out, gnsslog=args.gnsslog, obs=args.obs, nav=args.nav, nmea=args.nmea)


if __name__ == "__main__":
    main()
