"""Reusable single-epoch GNSS positioning core.

This is a refactor of the Ex0 ``positioning.py`` into composable pieces so the
anti-spoofing layer can reuse them:

  * ``compute_sat_states(epoch, nav_data, ...)`` turns an :class:`EpochObs`
    (source-agnostic measurements) into a list of :class:`SatState` objects,
    each carrying the satellite's rotation-corrected ECEF position/velocity,
    clock correction, the pseudorange and a measurement weight. SV geometry is
    computed **once per epoch** so RANSAC can re-solve over arbitrary subsets
    cheaply.
  * ``wls_position(sat_states, prior)`` runs the weighted least-squares PVT
    solve (receiver x, y, z, clock + one inter-system bias per non-GPS
    constellation) and returns a fix with residuals and DOP.
  * ``wls_velocity(...)`` solves receiver velocity + clock drift from Doppler.
  * ``solve_normal_equations`` / ``invert`` / ``compute_dop`` are exposed for
    the RAIM module (which needs the residual-sensitivity / hat matrix).

The maths (Keplerian propagation, GLONASS RK4, relativistic + TGD corrections,
earth-rotation correction) lives in :mod:`src.core.ephemeris` and is reused
verbatim from the Ex0 implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .coordinates import ecef_to_geodetic, elevation_azimuth
from .ephemeris import (
    C,
    apply_earth_rotation,
    compute_sv_ecef,
    propagate_glonass,
    sv_clock_correction,
)
from .measurements import EpochObs, RawMeasurement
from .rinex_nav import pick_ephemeris
from .timeutils import SECONDS_PER_WEEK, datetime_to_gps_sow

# ---------------------------------------------------------------------------
# Carrier frequencies (Hz) for the L1/E1/B1 signals phones typically track.
# Used to turn a Doppler / pseudorange-rate into a metric range rate.
# ---------------------------------------------------------------------------
F_L1_GPS = 1_575_420_000.0
F_E1_GAL = 1_575_420_000.0
F_B1I_BDS = 1_561_098_000.0
F_L1_GLO_BASE = 1_602_000_000.0
F_L1_GLO_STEP = 562_500.0

# Inter-system bias column order in the augmented design matrix (GPS = reference).
SYS_ORDER = ("E", "C", "R", "J")

GLONASS_LEAP_S = 18.0     # GPST - UTC, used to align GLONASS toc into GPS-time seconds
BDS_GPS_OFFSET = 14.0     # BDT is GPST - 14 s
SIGMA_FLOOR_M = 0.5       # clamp pseudorange uncertainty so one over-confident sat can't dominate

# GLONASS broadcast frame-time conventions vary between devices; a few seconds
# of mis-timing translates to tens of km of error, so it is opt-in.
SKIP_GLONASS = True

DEFAULT_PRIOR_XYZ = (4438000.0, 3086000.0, 3375000.0)  # rough Israel ECEF


@dataclass
class SatState:
    """Everything needed to form a pseudorange design row + residual for one SV.

    The predicted pseudorange for a receiver at ``rx`` with clock ``rx_clock``
    (metres) and inter-system bias ``isb`` is::

        rho = |sv_pos - rx|
        predicted = rho + rx_clock + isb - C * sv_clock_s
        residual  = pseudorange_m - predicted
    """

    sat_id: str
    sys: str
    prn: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    sv_clock_s: float
    pseudorange_m: float
    pr_rate_mps: Optional[float] = None
    cn0_dbhz: Optional[float] = None
    carrier_freq_hz: Optional[float] = None
    pr_uncertainty_m: Optional[float] = None
    weight: float = 1.0
    elevation_rad: Optional[float] = None
    azimuth_rad: Optional[float] = None

    @property
    def pos(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------

def _glonass_freq(channel_k: Optional[int]) -> float:
    if channel_k is None:
        return F_L1_GLO_BASE
    return F_L1_GLO_BASE + channel_k * F_L1_GLO_STEP


def carrier_freq_for(sat_id: str, glonass_slot_freq: Optional[float] = None) -> float:
    sys = sat_id[0]
    if sys == "G" or sys == "J":
        return F_L1_GPS
    if sys == "E":
        return F_E1_GAL
    if sys == "C":
        return F_B1I_BDS
    if sys == "R":
        return glonass_slot_freq or F_L1_GLO_BASE
    return F_L1_GPS


def _is_bds_geo(prn: int) -> bool:
    """BeiDou GEO satellites (handled poorly by the MEO formula) ~ PRN<=5 or >=59."""
    return prn <= 5 or prn >= 59


# ---------------------------------------------------------------------------
# Satellite state computation
# ---------------------------------------------------------------------------

def _compute_one_sat_state(
    meas: RawMeasurement,
    t_recv: float,
    rx_clock_m: float,
    week: int,
    nav_data,
    glonass_slot_to_k,
) -> Optional[Tuple[float, ...]]:
    """Return (x, y, z, vx, vy, vz, sv_clock_s) for one measurement, or None."""
    sys = meas.sys
    prn = meas.prn
    pr_m = meas.pseudorange_m

    if sys == "C" and _is_bds_geo(prn):
        return None
    if SKIP_GLONASS and sys == "R":
        return None

    # First transit-time estimate (light time + receiver clock).
    t_emit = t_recv - pr_m / C - rx_clock_m / C

    if sys == "R":
        eph_list = nav_data.get("R", {}).get(prn, [])
        eph = pick_ephemeris(eph_list, t_emit - GLONASS_LEAP_S, max_dt=2 * 3600)
        if eph is None:
            return None
        toc_secs = datetime_to_gps_sow(eph.toc, leap_seconds=0)
        dt = t_emit - GLONASS_LEAP_S - toc_secs
        state = propagate_glonass(eph, dt)
        sv_x, sv_y, sv_z = state[0], state[1], state[2]
        sv_vx, sv_vy, sv_vz = state[3], state[4], state[5]
        # The RINEX GLONASS clock field is already stored as -TauN, so the SV clock
        # correction is (-TauN) + GammaN*dt = tau_n + gamma_n*dt (do NOT re-negate).
        sv_clock = eph.tau_n + eph.gamma_n * dt
    else:
        offset = -BDS_GPS_OFFSET if sys == "C" else 0.0
        t_emit_sys = t_emit + offset
        t_emit_sys_sow = t_emit_sys - week * SECONDS_PER_WEEK
        eph_list = nav_data.get(sys, {}).get(prn, [])
        eph = pick_ephemeris(eph_list, t_emit_sys, max_dt=2 * 3600)
        if eph is None:
            return None
        sv_clock0 = sv_clock_correction(eph, t_emit_sys_sow)
        x, y, z, vx, vy, vz, rel = compute_sv_ecef(eph, t_emit_sys_sow - sv_clock0)
        sv_clock = sv_clock0 + rel - eph.tgd
        sv_x, sv_y, sv_z = x, y, z
        sv_vx, sv_vy, sv_vz = vx, vy, vz

    # Earth-rotation (Sagnac) correction over the signal transit time.
    transit = pr_m / C
    sx, sy, sz = apply_earth_rotation(sv_x, sv_y, sv_z, transit)
    return (sx, sy, sz, sv_vx, sv_vy, sv_vz, sv_clock)


def compute_sat_states(
    epoch: EpochObs,
    nav_data,
    glonass_slot_to_k: Optional[Dict[int, int]] = None,
    rx_prior=DEFAULT_PRIOR_XYZ,
    rx_clock_prior_m: float = 0.0,
    elevation_mask_rad: float = math.radians(10.0),
    apply_mask: bool = True,
) -> List[SatState]:
    """Build per-satellite states for one epoch (geometry computed once).

    ``rx_prior`` only needs to be good to the continent level — it is used for
    the emit-time light-time estimate, elevation masking and elevation
    weighting. The actual position is solved by :func:`wls_position`.
    """
    glonass_slot_to_k = glonass_slot_to_k or {}
    t_recv = datetime_to_gps_sow(epoch.time, leap_seconds=0)
    week = int(t_recv // SECONDS_PER_WEEK)

    states: List[SatState] = []
    have_real_prior = (rx_prior[0] ** 2 + rx_prior[1] ** 2 + rx_prior[2] ** 2) > 1e12

    for meas in epoch.sats:
        pr = meas.pseudorange_m
        if pr is None or pr <= 1e6 or pr > 5e7:
            continue
        sv = _compute_one_sat_state(
            meas, t_recv, rx_clock_prior_m, week, nav_data, glonass_slot_to_k
        )
        if sv is None:
            continue
        sx, sy, sz, vx, vy, vz, sv_clock = sv

        dx = sx - rx_prior[0]
        dy = sy - rx_prior[1]
        dz = sz - rx_prior[2]
        rho = math.sqrt(dx * dx + dy * dy + dz * dz)
        if rho < 1e3:
            continue

        el = az = None
        weight = 1.0
        if have_real_prior:
            el, az = elevation_azimuth(rx_prior, (sx, sy, sz))
            if apply_mask and el < elevation_mask_rad:
                continue
            weight = max(math.sin(el), 0.05) ** 2

        # If the source supplied a pseudorange uncertainty, fold it in (clamped to a
        # floor so a satellite reporting an implausibly tiny sigma can't dominate the WLS).
        if meas.pr_uncertainty_m is not None and meas.pr_uncertainty_m > 0:
            sigma = max(meas.pr_uncertainty_m, SIGMA_FLOOR_M)
            weight = weight / (sigma ** 2)

        cf = meas.carrier_freq_hz
        if cf is None:
            slot_k = glonass_slot_to_k.get(meas.prn) if meas.sys == "R" else None
            cf = carrier_freq_for(meas.sat_id, _glonass_freq(slot_k) if slot_k is not None else None)

        states.append(
            SatState(
                sat_id=meas.sat_id, sys=meas.sys, prn=meas.prn,
                x=sx, y=sy, z=sz, vx=vx, vy=vy, vz=vz,
                sv_clock_s=sv_clock,
                pseudorange_m=pr,
                pr_rate_mps=meas.pr_rate_mps,
                cn0_dbhz=meas.cn0_dbhz,
                carrier_freq_hz=cf,
                pr_uncertainty_m=meas.pr_uncertainty_m,
                weight=weight,
                elevation_rad=el, azimuth_rad=az,
            )
        )
    return states


# ---------------------------------------------------------------------------
# Weighted least-squares position
# ---------------------------------------------------------------------------

def _active_sys(states: List[SatState]) -> List[str]:
    seen = set(s.sys for s in states)
    return [s for s in SYS_ORDER if s in seen]


def wls_position(
    states: List[SatState],
    prior=DEFAULT_PRIOR_XYZ,
    prior_clock_m: float = 0.0,
    max_iter: int = 12,
    min_satellites: int = 4,
    compute_dop_flag: bool = True,
) -> Optional[Dict]:
    """Weighted least-squares PVT over the *given* satellite states.

    Unknowns: receiver ECEF (x, y, z), clock (c*dt), and one inter-system bias
    per non-GPS constellation present. Returns a fix dict or None on failure.
    """
    if len(states) < min_satellites:
        return None

    active = _active_sys(states)
    n_unk = 4 + len(active)
    if len(states) < n_unk:
        return None

    rx = list(prior)
    rx_clock = prior_clock_m
    isb = {s: 0.0 for s in active}
    converged = False
    residuals_final: List[float] = []

    def col_index(sys: str) -> Optional[int]:
        return 4 + active.index(sys) if sys in active else None

    for _ in range(max_iter):
        H: List[List[float]] = []
        res: List[float] = []
        w: List[float] = []
        for s in states:
            dx = s.x - rx[0]
            dy = s.y - rx[1]
            dz = s.z - rx[2]
            rho = math.sqrt(dx * dx + dy * dy + dz * dz)
            if rho < 1e3:
                continue
            ux, uy, uz = dx / rho, dy / rho, dz / rho
            extra = isb.get(s.sys, 0.0) if s.sys != "G" else 0.0
            predicted = rho + rx_clock + extra - C * s.sv_clock_s
            residual = s.pseudorange_m - predicted
            row = [-ux, -uy, -uz, 1.0] + [0.0] * len(active)
            ci = col_index(s.sys)
            if ci is not None:
                row[ci] = 1.0
            H.append(row)
            res.append(residual)
            w.append(s.weight)

        if len(H) < n_unk:
            return None
        try:
            dvec = solve_wls(H, res, w)
        except ZeroDivisionError:
            return None

        rx[0] += dvec[0]
        rx[1] += dvec[1]
        rx[2] += dvec[2]
        rx_clock += dvec[3]
        for i, sys in enumerate(active):
            isb[sys] += dvec[4 + i]

        residuals_final = res
        if all(abs(v) < 1e-3 for v in dvec[:3]):
            converged = True
            break

    if not converged:
        return None

    # Final post-fit residuals at the converged solution.
    per_sat_residual: Dict[str, float] = {}
    H_pos: List[List[float]] = []
    rss = 0.0
    sat_states_map: Dict[str, SatState] = {}
    for s in states:
        dx = s.x - rx[0]
        dy = s.y - rx[1]
        dz = s.z - rx[2]
        rho = math.sqrt(dx * dx + dy * dy + dz * dz)
        ux, uy, uz = dx / rho, dy / rho, dz / rho
        extra = isb.get(s.sys, 0.0) if s.sys != "G" else 0.0
        predicted = rho + rx_clock + extra - C * s.sv_clock_s
        r = s.pseudorange_m - predicted
        per_sat_residual[s.sat_id] = r
        rss += r * r
        H_pos.append([-ux, -uy, -uz, 1.0])
        sat_states_map[s.sat_id] = s

    n = len(states)
    rms_res = math.sqrt(rss / max(1, n))

    dop = {}
    if compute_dop_flag:
        try:
            dop = compute_dop(H_pos, *rx)
        except Exception:
            dop = {}

    return {
        "ecef": tuple(rx),
        "clock_bias_m": rx_clock,
        "isb": isb,
        "satellites": sat_states_map,
        "sat_ids": [s.sat_id for s in states],
        "residuals": per_sat_residual,
        "n_sats": n,
        "dop": dop,
        "rms_residual_m": rms_res,
    }


def wls_velocity(states: List[SatState], rx_xyz) -> Optional[Dict]:
    """Least-squares receiver velocity + clock drift from pseudorange rates."""
    rows: List[List[float]] = []
    obs: List[float] = []
    for s in states:
        if s.pr_rate_mps is None:
            continue
        dx = s.x - rx_xyz[0]
        dy = s.y - rx_xyz[1]
        dz = s.z - rx_xyz[2]
        rho = math.sqrt(dx * dx + dy * dy + dz * dz)
        if rho < 1e3:
            continue
        ux, uy, uz = dx / rho, dy / rho, dz / rho
        # pr_rate already in m/s (range rate). SV motion projected onto LOS.
        sv_proj = s.vx * ux + s.vy * uy + s.vz * uz
        y = s.pr_rate_mps - sv_proj
        rows.append([-ux, -uy, -uz, 1.0])
        obs.append(y)
    if len(rows) < 4:
        return None
    try:
        sol = solve_wls(rows, obs, [1.0] * len(rows))
    except ZeroDivisionError:
        return None
    return {"vx": sol[0], "vy": sol[1], "vz": sol[2], "clock_drift_m_s": sol[3]}


# ---------------------------------------------------------------------------
# Linear algebra (pure Python, no NumPy) — shared with RANSAC / RAIM
# ---------------------------------------------------------------------------

def solve_wls(H: List[List[float]], r: List[float], w: List[float]) -> List[float]:
    """Solve (Hᵀ W H) dx = Hᵀ W r and return dx."""
    n_obs = len(H)
    n_unk = len(H[0])
    HtH = [[0.0] * n_unk for _ in range(n_unk)]
    Htr = [0.0] * n_unk
    for i in range(n_obs):
        wi = w[i]
        Hi = H[i]
        ri = r[i]
        for j in range(n_unk):
            Htr[j] += Hi[j] * wi * ri
            HtHj = HtH[j]
            Hij_w = Hi[j] * wi
            for k in range(n_unk):
                HtHj[k] += Hij_w * Hi[k]
    return solve_linear(HtH, Htr)


def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        pivot = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[pivot][i]):
                pivot = k
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
        if abs(M[i][i]) < 1e-12:
            raise ZeroDivisionError("singular matrix")
        for k in range(i + 1, n):
            factor = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def invert(A: List[List[float]]) -> List[List[float]]:
    """Invert a square matrix via Gauss-Jordan."""
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for i in range(n):
        pivot = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[pivot][i]):
                pivot = k
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
        if abs(M[i][i]) < 1e-12:
            raise ZeroDivisionError("singular")
        inv_p = 1.0 / M[i][i]
        for j in range(2 * n):
            M[i][j] *= inv_p
        for k in range(n):
            if k == i:
                continue
            f = M[k][i]
            for j in range(2 * n):
                M[k][j] -= f * M[i][j]
    return [[M[i][j + n] for j in range(n)] for i in range(n)]


def compute_dop(H_pos: List[List[float]], x: float, y: float, z: float) -> Dict[str, float]:
    """DOPs from a 4-column position design matrix and the solved position."""
    HtH = [[0.0] * 4 for _ in range(4)]
    for row in H_pos:
        for i in range(4):
            for j in range(4):
                HtH[i][j] += row[i] * row[j]
    inv = invert(HtH)
    g = sum(inv[i][i] for i in range(4))
    p = sum(inv[i][i] for i in range(3))
    t = inv[3][3]
    lat, lon, _ = ecef_to_geodetic(x, y, z)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    R = [[-so, co, 0.0],
         [-sl * co, -sl * so, cl],
         [cl * co, cl * so, sl]]
    Q = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                for l in range(3):
                    acc += R[i][k] * inv[k][l] * R[j][l]
            Q[i][j] = acc
    h = Q[0][0] + Q[1][1]
    v = Q[2][2]
    return {
        "gdop": math.sqrt(max(g, 0.0)),
        "pdop": math.sqrt(max(p, 0.0)),
        "hdop": math.sqrt(max(h, 0.0)),
        "vdop": math.sqrt(max(v, 0.0)),
        "tdop": math.sqrt(max(t, 0.0)),
    }
