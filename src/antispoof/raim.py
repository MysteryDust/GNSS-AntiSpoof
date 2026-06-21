"""Receiver Autonomous Integrity Monitoring (RAIM).

Classical snapshot residual-based RAIM (Parkinson/Sturza family):

  z   = prefit pseudorange residuals               (n x 1)
  H   = augmented geometry matrix [-u | 1 | ISBs]  (n x m,  m = 4 + #non-GPS sys)
  W   = diag(1/sigma_i^2),  sigma_i = sigma_rho / sin(el_i)
  dx  = (HᵀWH)⁻¹ HᵀW z
  w   = z - H dx                                   (postfit residuals)
  SSE = wᵀ W w   ~  chi²(dof),  dof = n - m   under the no-fault hypothesis

Fault Detection (FD): declare a fault if SSE > T = chi2.isf(Pfa, dof).
Fault Detection & Exclusion (FDE): iteratively drop the satellite with the
largest *normalised* residual |w_i|·sqrt(W_ii)/sqrt(S_ii) (S = I − hat matrix)
and re-test, until SSE passes or redundancy runs out.

Classical RAIM detects a *single* fault well but, as the literature (and our
own simulator) shows, it mislabels the minority when several satellites are
spoofed consistently. RANSAC (``ransac.py``) handles the subset isolation; RAIM
here provides (a) the formal integrity test statistic / protection-level style
flag, and (b) a self-consistency check on whichever satellite set RANSAC keeps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.ephemeris import C
from ..core.solver import SatState, SYS_ORDER, invert, solve_wls
from .stats import chi2_isf

DEFAULT_SIGMA_RHO = 5.0     # [m] nominal pseudorange sigma at zenith
DEFAULT_PFA = 1e-5          # per-epoch false-alarm probability


@dataclass
class RaimResult:
    ok: bool                                   # enough redundancy to run a test
    fault_detected: bool = False
    sse: float = 0.0
    threshold: float = 0.0
    dof: int = 0
    n_sats: int = 0
    excluded_ids: List[str] = field(default_factory=list)
    normalized_residuals: Dict[str, float] = field(default_factory=dict)
    postfit_residuals: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


def _active_sys(states: List[SatState]) -> List[str]:
    seen = set(s.sys for s in states)
    return [s for s in SYS_ORDER if s in seen]


def _sigma(state: SatState, sigma_rho: float) -> float:
    """Elevation-dependent measurement sigma (metres)."""
    if state.pr_uncertainty_m and state.pr_uncertainty_m > 0:
        base = state.pr_uncertainty_m
    else:
        base = sigma_rho
    if state.elevation_rad is not None:
        return base / max(math.sin(state.elevation_rad), 0.1)
    return base


def _design_and_residuals(states: List[SatState], rx, rx_clock_m: float,
                          isb: Dict[str, float], active: List[str]):
    """Build the augmented design matrix, prefit residuals and weights."""
    H: List[List[float]] = []
    z: List[float] = []
    w: List[float] = []
    sigmas: List[float] = []
    for s in states:
        dx = s.x - rx[0]; dy = s.y - rx[1]; dz = s.z - rx[2]
        rho = math.sqrt(dx * dx + dy * dy + dz * dz)
        ux, uy, uz = dx / rho, dy / rho, dz / rho
        extra = isb.get(s.sys, 0.0) if s.sys != "G" else 0.0
        predicted = rho + rx_clock_m + extra - C * s.sv_clock_s
        row = [-ux, -uy, -uz, 1.0] + [0.0] * len(active)
        if s.sys in active:
            row[4 + active.index(s.sys)] = 1.0
        sig = _sigma(s, DEFAULT_SIGMA_RHO)
        H.append(row)
        z.append(s.pseudorange_m - predicted)
        w.append(1.0 / (sig * sig))
        sigmas.append(sig)
    return H, z, w, sigmas


def _solve_residuals(states: List[SatState], sigma_rho: float):
    """One WLS solve about a prior; return postfit residuals, weights, SSE, dof, S_ii.

    The solve is seeded from a robust internal estimate so RAIM does not depend
    on an external fix being supplied.
    """
    active = _active_sys(states)
    m = 4 + len(active)
    n = len(states)
    if n <= m:
        return None  # no redundancy

    # Seed position/clock from a quick WLS about the centroid-of-LOS prior.
    rx = [states[0].x * 0.0, 0.0, 0.0]  # placeholder; iterate from Earth centre is bad
    # Better prior: average satellite direction scaled to Earth radius is unreliable,
    # so use the mean of a coarse single-point solve via normal iteration from a
    # rough Earth-surface guess derived from the first satellite sub-point.
    s0 = states[0]
    norm0 = math.sqrt(s0.x ** 2 + s0.y ** 2 + s0.z ** 2)
    rx = [s0.x / norm0 * 6371000.0, s0.y / norm0 * 6371000.0, s0.z / norm0 * 6371000.0]
    rx_clock = 0.0
    isb = {s: 0.0 for s in active}

    for _ in range(15):
        H, z, w, sigmas = _design_and_residuals(states, rx, rx_clock, isb, active)
        try:
            dvec = solve_wls(H, z, w)
        except ZeroDivisionError:
            return None
        rx[0] += dvec[0]; rx[1] += dvec[1]; rx[2] += dvec[2]
        rx_clock += dvec[3]
        for i, sy in enumerate(active):
            isb[sy] += dvec[4 + i]
        if all(abs(v) < 1e-3 for v in dvec[:3]):
            break

    # Postfit residuals at the converged solution.
    H, z, w, sigmas = _design_and_residuals(states, rx, rx_clock, isb, active)
    # Solve once more to get postfit residual vector w_res = z - H dx (dx ~ 0 here).
    try:
        dvec = solve_wls(H, z, w)
    except ZeroDivisionError:
        return None
    w_res = []
    for i, row in enumerate(H):
        pred = sum(row[j] * dvec[j] for j in range(len(row)))
        w_res.append(z[i] - pred)

    sse = sum(w[i] * w_res[i] * w_res[i] for i in range(n))
    dof = n - m

    # Hat-matrix diagonal for normalised residuals: P_ii = W_ii * h_iᵀ M h_i,
    # M = (HᵀWH)⁻¹,  S_ii = 1 - P_ii.
    n_unk = len(H[0])
    HtWH = [[0.0] * n_unk for _ in range(n_unk)]
    for i in range(n):
        wi = w[i]
        Hi = H[i]
        for a in range(n_unk):
            for b in range(n_unk):
                HtWH[a][b] += Hi[a] * wi * Hi[b]
    try:
        M = invert(HtWH)
    except ZeroDivisionError:
        return None
    s_ii = []
    for i in range(n):
        Hi = H[i]
        quad = 0.0
        for a in range(n_unk):
            ma = 0.0
            for b in range(n_unk):
                ma += M[a][b] * Hi[b]
            quad += Hi[a] * ma
        p_ii = w[i] * quad
        s_ii.append(max(1.0 - p_ii, 1e-6))

    return {
        "rx": tuple(rx), "rx_clock": rx_clock, "isb": isb,
        "w_res": w_res, "weights": w, "sse": sse, "dof": dof,
        "s_ii": s_ii, "n": n, "m": m,
    }


def raim_fde(
    states: List[SatState],
    sigma_rho: float = DEFAULT_SIGMA_RHO,
    pfa: float = DEFAULT_PFA,
    max_exclude: int = 4,
) -> RaimResult:
    """Run RAIM fault detection and (if redundancy allows) exclusion."""
    working = list(states)
    excluded: List[str] = []

    sol = _solve_residuals(working, sigma_rho)
    if sol is None:
        return RaimResult(ok=False, n_sats=len(states),
                          reason="insufficient redundancy for RAIM (need n > unknowns)")

    while True:
        dof = sol["dof"]
        sse = sol["sse"]
        T = chi2_isf(pfa, dof) if dof >= 1 else float("inf")
        norm_res = {}
        for i, s in enumerate(working):
            norm_res[s.sat_id] = abs(sol["w_res"][i]) * math.sqrt(sol["weights"][i]) / math.sqrt(sol["s_ii"][i])
        postfit = {working[i].sat_id: sol["w_res"][i] for i in range(len(working))}

        if sse <= T or dof < 1:
            return RaimResult(
                ok=True, fault_detected=bool(excluded), sse=sse, threshold=T, dof=dof,
                n_sats=len(working), excluded_ids=excluded,
                normalized_residuals=norm_res, postfit_residuals=postfit,
                reason="consistent" if not excluded else "consistent after exclusion",
            )

        # Fault detected. Try to exclude the satellite with the largest normalised
        # residual — but only if the *post-removal* set keeps redundancy (the unknown
        # count can drop if we remove the last satellite of a constellation).
        worst = max(norm_res, key=norm_res.get)
        candidate = [s for s in working if s.sat_id != worst]
        m_post = 4 + len(_active_sys(candidate))
        if len(excluded) >= max_exclude or len(candidate) <= m_post:
            # Cannot exclude further: report detection on the CURRENT (valid) set.
            return RaimResult(
                ok=True, fault_detected=True, sse=sse, threshold=T, dof=dof,
                n_sats=len(working), excluded_ids=excluded,
                normalized_residuals=norm_res, postfit_residuals=postfit,
                reason="fault detected; insufficient redundancy to exclude further",
            )
        new_sol = _solve_residuals(candidate, sigma_rho)
        if new_sol is None:
            return RaimResult(
                ok=True, fault_detected=True, sse=sse, threshold=T, dof=dof,
                n_sats=len(working), excluded_ids=excluded,
                normalized_residuals=norm_res, postfit_residuals=postfit,
                reason="fault detected; cannot exclude further (solve failed)",
            )
        excluded.append(worst)
        working = candidate
        sol = new_sol


def raim_consistency(states: List[SatState], sigma_rho: float = DEFAULT_SIGMA_RHO,
                     pfa: float = DEFAULT_PFA) -> Tuple[bool, float, float, int]:
    """Lightweight self-consistency test for a given satellite set.

    Returns (is_consistent, sse, threshold, dof). Used by the engine to confirm
    that the RANSAC inlier set is internally consistent (SSE below threshold).
    """
    sol = _solve_residuals(list(states), sigma_rho)
    if sol is None:
        return True, 0.0, float("inf"), 0
    dof = sol["dof"]
    T = chi2_isf(pfa, dof) if dof >= 1 else float("inf")
    return sol["sse"] <= T, sol["sse"], T, dof
