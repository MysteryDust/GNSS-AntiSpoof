"""Synthetic GNSS scenario generator with ground truth.

To prove the anti-spoofing engine quantitatively we need a scenario where the
*true* receiver position is known and a chosen *subset* of satellites can be
spoofed on demand. This module builds exactly that, fully offline and
deterministically:

  * ``make_synthetic_constellation`` creates broadcast Keplerian ephemerides for
    a configurable GPS + Galileo constellation (same ``KeplerEph`` records the
    real RINEX parser produces, so the rest of the pipeline is none the wiser).
  * ``simulate_clean_epoch`` produces a clean :class:`EpochObs` for a known
    receiver ECEF position by running the *forward* measurement model — and it
    reuses the solver's own ``_compute_one_sat_state`` so the generated
    pseudoranges are, by construction, invertible by the WLS solver to within
    the injected noise.
  * ``simulate_track`` walks a trajectory and returns the epoch stream plus the
    ground-truth positions for scoring.

A configurable receiver inter-system bias (e.g. +12 m on Galileo) is injected so
the ISB estimation path is genuinely exercised, and elevation-dependent C/N0 is
attached so the physical-layer monitor has realistic values to watch.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from ..core.coordinates import elevation_azimuth, geodetic_to_ecef
from ..core.ephemeris import C, OMEGA_E
from ..core.measurements import EpochObs, RawMeasurement
from ..core.rinex_nav import KeplerEph
from ..core.solver import _compute_one_sat_state
from ..core.timeutils import SECONDS_PER_WEEK, datetime_to_gps_sow

MU_GPS = 3.986005e14
GPS_A = 26_560_000.0           # GPS semi-major axis (m)
GAL_A = 29_600_000.0           # Galileo semi-major axis (m)


@dataclass
class TruthSample:
    time: datetime
    ecef: Tuple[float, float, float]
    lat_deg: float
    lon_deg: float
    alt_m: float
    clock_bias_m: float


def _make_kepler(sys: str, prn: int, a: float, inc_rad: float, raan_rad: float,
                 m0_rad: float, toc: datetime, toe_sow: float, gnss_week: int,
                 af0: float = 0.0, af1: float = 0.0, ecc: float = 0.001) -> KeplerEph:
    """Build a minimal but valid Keplerian ephemeris (harmonic terms zeroed)."""
    return KeplerEph(
        sys=sys, prn=prn, toc=toc, af0=af0, af1=af1, af2=0.0,
        iode=0.0, crs=0.0, delta_n=0.0, m0=m0_rad,
        cuc=0.0, e=ecc, cus=0.0, sqrt_a=math.sqrt(a),
        toe=toe_sow, cic=0.0, omega0=raan_rad, cis=0.0,
        i0=inc_rad, crc=0.0, omega=0.0, omega_dot=0.0,
        idot=0.0, codes_l2=0.0, gnss_week=float(gnss_week), l2p_flag=0.0,
        sv_accuracy=2.0, sv_health=0.0, tgd=0.0, iodc_or_bgd_e5a_e1=0.0,
        transmission_time=toe_sow, fit_interval=4.0, spare1=0.0, spare2=0.0,
    )


BDS_A = 27_906_000.0           # BeiDou MEO semi-major axis (m)


def make_synthetic_constellation(
    ref_time: Optional[datetime] = None,
    n_gps: int = 24,
    n_gal: int = 18,
    n_bds: int = 0,
    seed: int = 7,
) -> Tuple[Dict, datetime]:
    """Return (nav_data, ref_time) for a synthetic GPS+Galileo constellation.

    Satellites are laid out as a Walker-style constellation (several equally
    spaced orbital planes, satellites phased within each plane) so that a ground
    receiver sees ~8-14 satellites above the elevation mask with good geometry.
    ``nav_data`` matches the structure the real RINEX navigation parser yields.
    """
    if ref_time is None:
        ref_time = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)
    rng = random.Random(seed)
    toe_abs = datetime_to_gps_sow(ref_time, leap_seconds=0)
    gnss_week = int(toe_abs // SECONDS_PER_WEEK)
    toe_sow = toe_abs - gnss_week * SECONDS_PER_WEEK

    nav: Dict[str, Dict[int, List[KeplerEph]]] = {"G": {}, "E": {}, "C": {}, "R": {}, "J": {}}

    def populate(sys: str, n: int, a: float, inc_deg: float, planes: int, prn_start: int = 1):
        inc = math.radians(inc_deg)
        per_plane = max(1, round(n / planes))
        prn = prn_start
        last = prn_start + n - 1
        for p in range(planes):
            raan = 2.0 * math.pi * p / planes
            for s in range(per_plane):
                if prn > last:
                    break
                # phase satellites within the plane, with a per-plane offset so
                # adjacent planes are interleaved (Walker F phasing).
                m0 = 2.0 * math.pi * (s / per_plane) + (2.0 * math.pi * p / (planes * per_plane))
                m0 += rng.uniform(-0.05, 0.05)
                af0 = rng.uniform(-1e-5, 1e-5)        # clock bias (s); *C ~ few km, solver handles
                af1 = rng.uniform(-1e-12, 1e-12)
                nav[sys].setdefault(prn, []).append(
                    _make_kepler(sys, prn, a, inc, raan, m0, ref_time, toe_sow, gnss_week,
                                 af0=af0, af1=af1, ecc=rng.uniform(0.0005, 0.008))
                )
                prn += 1

    populate("G", n_gps, GPS_A, 55.0, planes=6)
    populate("E", n_gal, GAL_A, 56.0, planes=3)
    if n_bds:
        # BeiDou MEO PRNs (avoid the GEO range the solver skips: prn<=5 or >=59).
        populate("C", n_bds, BDS_A, 55.0, planes=3, prn_start=19)
    return nav, ref_time


def _forward_pseudorange(meas_template: RawMeasurement, rx, t_recv: float,
                         rx_clock_m: float, week: int, nav_data, slots) -> Optional[Tuple]:
    """Fixed-point solve of the forward pseudorange using the solver's SV model.

    Returns (pseudorange_m, sv_xyz, sv_vel, sv_clock_s, elevation_rad) or None.
    """
    pr = 2.2e7  # initial guess (~ LEO-to-MEO range scale)
    sv = None
    for _ in range(6):
        meas_template.pseudorange_m = pr
        sv = _compute_one_sat_state(meas_template, t_recv, rx_clock_m, week, nav_data, slots)
        if sv is None:
            return None
        sx, sy, sz, vx, vy, vz, sv_clock = sv
        dx = sx - rx[0]; dy = sy - rx[1]; dz = sz - rx[2]
        geom = math.sqrt(dx * dx + dy * dy + dz * dz)
        new_pr = geom + rx_clock_m - C * sv_clock
        if abs(new_pr - pr) < 1e-4:
            pr = new_pr
            break
        pr = new_pr
    sx, sy, sz, vx, vy, vz, sv_clock = sv
    el, _az = elevation_azimuth(rx, (sx, sy, sz))
    return pr, (sx, sy, sz), (vx, vy, vz), sv_clock, el


def _cn0_from_elevation(el_rad: float, rng: random.Random) -> float:
    """A simple but realistic C/N0 vs elevation curve (open-sky)."""
    el_deg = math.degrees(el_rad)
    base = 33.0 + 0.30 * min(el_deg, 60.0)         # ~33 dB-Hz at horizon -> ~51 near zenith
    return base + rng.gauss(0.0, 0.8)


def simulate_clean_epoch(
    rx_ecef: Tuple[float, float, float],
    rx_vel: Tuple[float, float, float],
    t: datetime,
    nav_data,
    rx_clock_m: float = 0.0,
    rx_clock_drift_mps: float = 0.0,
    isb_true: Optional[Dict[str, float]] = None,
    elevation_mask_deg: float = 10.0,
    noise_m: float = 0.6,
    seed: int = 0,
) -> EpochObs:
    """Generate one clean epoch of measurements for a known receiver state."""
    isb_true = isb_true or {}
    rng = random.Random((seed, int(datetime_to_gps_sow(t, 0))))
    t_recv = datetime_to_gps_sow(t, leap_seconds=0)
    week = int(t_recv // SECONDS_PER_WEEK)
    mask = math.radians(elevation_mask_deg)
    slots: Dict[int, int] = {}

    sats: List[RawMeasurement] = []
    agc_bands: Dict[str, float] = {}
    for sys, prns in nav_data.items():
        if sys not in ("G", "E", "C", "J"):
            continue
        for prn in prns:
            sat_id = f"{sys}{prn:02d}"
            template = RawMeasurement(sat_id=sat_id, sys=sys, prn=prn, pseudorange_m=2.2e7)
            fwd = _forward_pseudorange(template, rx_ecef, t_recv, rx_clock_m, week, nav_data, slots)
            if fwd is None:
                continue
            pr, sv_xyz, sv_vel, sv_clock, el = fwd
            if el < mask:
                continue
            # Receiver inter-system bias + measurement noise.
            pr_meas = pr + isb_true.get(sys, 0.0) + rng.gauss(0.0, noise_m)
            # Range rate: u . (v_sv - v_rx) + clock drift.
            dx = sv_xyz[0] - rx_ecef[0]; dy = sv_xyz[1] - rx_ecef[1]; dz = sv_xyz[2] - rx_ecef[2]
            rho = math.sqrt(dx * dx + dy * dy + dz * dz)
            ux, uy, uz = dx / rho, dy / rho, dz / rho
            rel_vx = sv_vel[0] - rx_vel[0]
            rel_vy = sv_vel[1] - rx_vel[1]
            rel_vz = sv_vel[2] - rx_vel[2]
            pr_rate = (ux * rel_vx + uy * rel_vy + uz * rel_vz) + rx_clock_drift_mps
            pr_rate += rng.gauss(0.0, 0.03)
            cn0 = _cn0_from_elevation(el, rng)
            sats.append(
                RawMeasurement(
                    sat_id=sat_id, sys=sys, prn=prn,
                    pseudorange_m=pr_meas, pr_rate_mps=pr_rate,
                    cn0_dbhz=cn0, carrier_freq_hz=None,
                    pr_uncertainty_m=noise_m, code="SIM",
                )
            )

    # Clean AGC baseline per band (arbitrary nominal level in dB).
    for band in ("L1",):
        agc_bands[band] = 40.0 + rng.gauss(0.0, 0.2)

    return EpochObs(time=t, sats=sats, agc_db=agc_bands)


def make_trajectory(
    start_lat: float = 32.103,
    start_lon: float = 34.808,
    start_alt: float = 40.0,
    n_epochs: int = 120,
    dt_s: float = 1.0,
    speed_mps: float = 6.0,
    heading_deg: float = 90.0,
    ref_time: Optional[datetime] = None,
    clock_bias_m: float = 30.0,
    clock_drift_mps: float = 0.1,
) -> List[TruthSample]:
    """A constant-velocity ground track (default: ~Tel Aviv, moving east at 6 m/s)."""
    if ref_time is None:
        ref_time = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)
    samples: List[TruthSample] = []
    lat = start_lat
    lon = start_lon
    heading = math.radians(heading_deg)
    # metres per degree at this latitude
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(start_lat))
    for i in range(n_epochs):
        t = ref_time + timedelta(seconds=i * dt_s)
        dist = speed_mps * dt_s
        dn = dist * math.cos(heading)
        de = dist * math.sin(heading)
        lat += (dn / m_per_deg_lat) * (1 if i > 0 else 0)
        lon += (de / m_per_deg_lon) * (1 if i > 0 else 0)
        x, y, z = geodetic_to_ecef(math.radians(lat), math.radians(lon), start_alt)
        samples.append(
            TruthSample(time=t, ecef=(x, y, z), lat_deg=lat, lon_deg=lon,
                        alt_m=start_alt, clock_bias_m=clock_bias_m + clock_drift_mps * i)
        )
    return samples


def velocity_enu_to_ecef(speed_mps: float, heading_deg: float, lat_deg: float, lon_deg: float):
    """Convert a horizontal speed/heading into an ECEF velocity vector."""
    heading = math.radians(heading_deg)
    vn = speed_mps * math.cos(heading)
    ve = speed_mps * math.sin(heading)
    vu = 0.0
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    # ENU -> ECEF
    vx = -so * ve - sl * co * vn + cl * co * vu
    vy = co * ve - sl * so * vn + cl * so * vu
    vz = cl * vn + sl * vu
    return (vx, vy, vz)


def simulate_track(
    trajectory: List[TruthSample],
    nav_data,
    isb_true: Optional[Dict[str, float]] = None,
    speed_mps: float = 6.0,
    heading_deg: float = 90.0,
    clock_drift_mps: float = 0.1,
    noise_m: float = 0.6,
    seed: int = 0,
) -> List[EpochObs]:
    """Generate the clean epoch stream for a whole trajectory."""
    epochs: List[EpochObs] = []
    for ts in trajectory:
        rx_vel = velocity_enu_to_ecef(speed_mps, heading_deg, ts.lat_deg, ts.lon_deg)
        ep = simulate_clean_epoch(
            ts.ecef, rx_vel, ts.time, nav_data,
            rx_clock_m=ts.clock_bias_m, rx_clock_drift_mps=clock_drift_mps,
            isb_true=isb_true, noise_m=noise_m, seed=seed,
        )
        epochs.append(ep)
    return epochs
