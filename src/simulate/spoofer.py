"""Inject a controllable spoofing attack on a *subset* of satellites.

Given a clean simulated scenario (epochs + ground-truth trajectory + the same
broadcast ephemeris used to generate it), this replaces the pseudoranges of a
chosen subset of satellites with values that are self-consistent with a *false*
receiver position. That is exactly how a single-antenna meaconer/simulator
attack looks to the receiver: the spoofed subset agrees with each other (and
with the spoofer's intended position) while the authentic satellites still agree
with the true position.

The attack also leaves the physical-layer fingerprints the literature describes:
the spoofed satellites' C/N0 is driven abnormally high and uniform, and the
epoch AGC is dropped (added RF power) for the duration of the attack.

What is returned alongside the spoofed epochs is the ground-truth attack
metadata (which PRNs were spoofed each epoch, and the false target position) so
detection can be scored precisely.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.coordinates import geodetic_to_ecef, ecef_to_geodetic
from ..core.measurements import EpochObs, RawMeasurement
from .scenario import TruthSample, _forward_pseudorange
from ..core.timeutils import datetime_to_gps_sow, SECONDS_PER_WEEK


@dataclass
class SpoofConfig:
    target_prns: List[str]                       # satellites to spoof, e.g. ["G01","G05","E03"]
    start_epoch: int                             # first epoch index of the attack
    end_epoch: int                               # last epoch index (inclusive)
    false_offset_enu: Tuple[float, float, float] = (600.0, 250.0, 0.0)  # E,N,U metres from truth
    ramp: bool = True                            # ramp the false target from truth -> full offset
    cn0_boost_to: float = 50.0                   # spoofed sats' C/N0 (dB-Hz)
    cn0_jitter: float = 0.4                      # small spread around the boost
    agc_drop_db: float = 8.0                     # epoch AGC drop while attacking
    noise_m: float = 0.4
    seed: int = 99


@dataclass
class AttackMeta:
    spoofed_prns_by_epoch: List[List[str]] = field(default_factory=list)
    false_pos_by_epoch: List[Optional[Tuple[float, float, float]]] = field(default_factory=list)
    active_by_epoch: List[bool] = field(default_factory=list)


def _enu_offset_to_ecef(lat_deg: float, lon_deg: float, e: float, n: float, u: float):
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    dx = -so * e - sl * co * n + cl * co * u
    dy = co * e - sl * so * n + cl * so * u
    dz = cl * n + sl * u
    return dx, dy, dz


def apply_spoofing(
    epochs: List[EpochObs],
    truth: List[TruthSample],
    nav_data,
    cfg: SpoofConfig,
    isb_true: Optional[Dict[str, float]] = None,
) -> Tuple[List[EpochObs], AttackMeta]:
    """Return (spoofed_epochs, attack_meta). The input epochs are not mutated."""
    isb_true = isb_true or {}
    rng = random.Random(cfg.seed)
    targets = set(cfg.target_prns)
    meta = AttackMeta()
    out: List[EpochObs] = []
    slots: Dict[int, int] = {}

    for i, ep in enumerate(epochs):
        active = cfg.start_epoch <= i <= cfg.end_epoch
        ts = truth[i]
        false_pos = None
        spoofed_here: List[str] = []

        if active:
            # Ramp the false target so the attack "walks" the solution off truth.
            frac = 1.0
            if cfg.ramp and cfg.end_epoch > cfg.start_epoch:
                frac = (i - cfg.start_epoch + 1) / (cfg.end_epoch - cfg.start_epoch + 1)
            e, n, u = (c * frac for c in cfg.false_offset_enu)
            dx, dy, dz = _enu_offset_to_ecef(ts.lat_deg, ts.lon_deg, e, n, u)
            false_pos = (ts.ecef[0] + dx, ts.ecef[1] + dy, ts.ecef[2] + dz)

        t_recv = datetime_to_gps_sow(ep.time, leap_seconds=0)
        week = int(t_recv // SECONDS_PER_WEEK)

        new_sats: List[RawMeasurement] = []
        for m in ep.sats:
            if active and m.sat_id in targets and false_pos is not None:
                # Recompute this satellite's pseudorange as if the receiver were
                # at the false position (same clock as truth -> consistent subset).
                template = RawMeasurement(sat_id=m.sat_id, sys=m.sys, prn=m.prn, pseudorange_m=2.2e7)
                fwd = _forward_pseudorange(template, false_pos, t_recv, ts.clock_bias_m,
                                           week, nav_data, slots)
                if fwd is None:
                    new_sats.append(m)
                    continue
                pr_false, sv_xyz, sv_vel, sv_clock, el = fwd
                pr_false += isb_true.get(m.sys, 0.0) + rng.gauss(0.0, cfg.noise_m)
                cn0 = cfg.cn0_boost_to + rng.gauss(0.0, cfg.cn0_jitter)
                new_sats.append(
                    RawMeasurement(
                        sat_id=m.sat_id, sys=m.sys, prn=m.prn,
                        pseudorange_m=pr_false, pr_rate_mps=m.pr_rate_mps,
                        cn0_dbhz=cn0, carrier_freq_hz=m.carrier_freq_hz,
                        pr_uncertainty_m=m.pr_uncertainty_m, code="SPOOFED",
                    )
                )
                spoofed_here.append(m.sat_id)
            else:
                new_sats.append(m)

        agc = dict(ep.agc_db)
        if active and spoofed_here:
            for band in agc:
                agc[band] = agc[band] - cfg.agc_drop_db

        out.append(EpochObs(time=ep.time, sats=new_sats, agc_db=agc))
        meta.spoofed_prns_by_epoch.append(spoofed_here)
        meta.false_pos_by_epoch.append(false_pos)
        meta.active_by_epoch.append(active and bool(spoofed_here))

    return out, meta
