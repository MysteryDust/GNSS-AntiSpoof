"""RANSAC pseudorange consensus — isolate a spoofed *subset* of satellites.

The brief asks us to detect and overcome spoofing of a *subset* of the
navigation satellites. RANSAC (Random Sample Consensus, Fischler & Bolles 1981)
is the natural tool: it fits a PVT model to many minimal satellite subsets and
keeps the position supported by the largest mutually-consistent (consensus) set,
treating the rest as outliers. The spoofed subset shows up as the excluded set.

Design choices (grounded in P-RANSAC / Castaldo 2014 and Zhu TDCP RANSAC 2022):

* **Each candidate solves the full multi-constellation model itself.** The
  minimal subset has ``4 + (#non-GPS constellations present)`` satellites and is
  required to *cover* every constellation present, so each hypothesis estimates
  its own inter-system biases. This avoids a single global ISB estimate, which a
  spoofed satellite would corrupt — poisoning the consensus for every other
  satellite of that constellation.
* **Geometry pre-screening.** A degenerate minimal subset (nearly coplanar lines
  of sight) yields an ill-conditioned solve and a wild candidate position that
  can accumulate a false consensus. We reject any subset whose solution PDOP
  exceeds ``pdop_max``.
* **Exhaustive when cheap, random when not.** If C(m,s) is small we enumerate all
  covering subsets; otherwise we sample N = ⌈log(1−p)/log(1−wˢ)⌉ subsets with a
  seeded RNG and shrink N online as the best consensus grows.
* **Elevation-weighted inlier gate** k·σ/ sin(el).
* **Trust anchor.** RANSAC cannot by itself survive a *majority* spoof (it would
  lock onto the spoofed consensus). The caller may pass a ``position_prior`` and
  ``gate_m``; a consensus whose position jumps further than the gate from the
  prior is rejected and reported, so the engine can fall back / raise an alarm.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Dict, List, Optional, Tuple

from ..core.ephemeris import C
from ..core.solver import SatState, SYS_ORDER, wls_position

DEFAULT_P = 0.99            # desired probability of hitting a clean subset
DEFAULT_W0 = 0.6           # conservative initial inlier-ratio guess
DEFAULT_K_SIGMA = 4.0      # inlier-gate multiplier (3-5 typical)
DEFAULT_SIGMA_RHO = 5.0    # [m] nominal single-frequency pseudorange noise
DEFAULT_PDOP_MAX = 8.0     # reject minimal subsets with worse geometry than this
ENUMERATE_CAP = 800        # enumerate all covering subsets up to this many
SAMPLE_CAP = 400           # hard cap on random hypotheses per epoch


@dataclass
class RansacResult:
    ok: bool
    fix: Optional[Dict]                       # final WLS fix over the inlier set
    inlier_ids: List[str] = field(default_factory=list)
    excluded_ids: List[str] = field(default_factory=list)
    residuals: Dict[str, float] = field(default_factory=dict)   # post-fit residual per sat (m)
    thresholds: Dict[str, float] = field(default_factory=dict)  # per-sat gate (m)
    consensus_size: int = 0
    iterations: int = 0
    prior_gate_rejected: bool = False         # consensus jumped beyond position prior
    reason: str = ""


def ransac_iterations(p: float, w: float, s: int) -> int:
    w = min(max(w, 1e-3), 0.999)
    denom = math.log(1.0 - w ** s)
    if denom == 0.0:
        return 1
    return max(1, math.ceil(math.log(1.0 - p) / denom))


def _active_sys(states: List[SatState]) -> List[str]:
    seen = set(s.sys for s in states)
    return [s for s in SYS_ORDER if s in seen]


def _predict_residual(state: SatState, ecef, clock_m: float, isb: Dict[str, float]) -> float:
    dx = state.x - ecef[0]
    dy = state.y - ecef[1]
    dz = state.z - ecef[2]
    rho = math.sqrt(dx * dx + dy * dy + dz * dz)
    extra = isb.get(state.sys, 0.0) if state.sys != "G" else 0.0
    predicted = rho + clock_m + extra - C * state.sv_clock_s
    return state.pseudorange_m - predicted


def _gate(state: SatState, k_sigma: float, sigma_rho: float) -> float:
    if state.elevation_rad is not None:
        sin_el = max(math.sin(state.elevation_rad), 0.1)
    else:
        sin_el = 1.0
    return k_sigma * sigma_rho / sin_el


def _covers(subset: List[SatState], active: List[str]) -> bool:
    """True if the subset contains at least one satellite of every non-GPS
    constellation present, so all inter-system biases are estimable."""
    present = set(s.sys for s in subset)
    return all(a in present for a in active)


def ransac_pvt(
    states: List[SatState],
    prior=None,
    prior_clock_m: float = 0.0,
    p: float = DEFAULT_P,
    w0: float = DEFAULT_W0,
    k_sigma: float = DEFAULT_K_SIGMA,
    sigma_rho: float = DEFAULT_SIGMA_RHO,
    pdop_max: float = DEFAULT_PDOP_MAX,
    min_consensus: Optional[int] = None,
    position_prior=None,
    gate_m: Optional[float] = None,
    seed: int = 12345,
) -> RansacResult:
    """Run RANSAC consensus PVT over the epoch's satellite states."""
    m = len(states)
    active = _active_sys(states)
    s = 4 + len(active)               # minimal subset size (covers all ISBs)
    if m < s + 1:
        return RansacResult(ok=False, fix=None, reason=f"too few satellites ({m}) for {s}+1")

    if prior is None:
        prior = (states[0].x, states[0].y, states[0].z)
    if min_consensus is None:
        min_consensus = s + 1

    gates = [_gate(st, k_sigma, sigma_rho) for st in states]
    rng = random.Random(seed)

    total = comb(m, s)
    exhaustive = total <= ENUMERATE_CAP
    if exhaustive:
        candidates = [idx for idx in combinations(range(m), s)
                      if _covers([states[i] for i in idx], active)]
        max_iter = len(candidates)
    else:
        candidates = None
        max_iter = min(SAMPLE_CAP, ransac_iterations(p, w0, s))

    best_inliers: List[int] = []
    best_cost = float("inf")
    iters_done = 0

    def evaluate(idx: Tuple[int, ...]):
        subset = [states[j] for j in idx]
        if not _covers(subset, active):
            return None
        fix = wls_position(subset, prior=prior, prior_clock_m=prior_clock_m,
                           min_satellites=s, compute_dop_flag=True)
        if fix is None:
            return None
        if fix.get("dop", {}).get("pdop", 999.0) > pdop_max:
            return None                       # geometry pre-screen
        ecef = fix["ecef"]
        clk = fix["clock_bias_m"]
        isb = fix.get("isb", {})
        inliers = []
        cost = 0.0
        for j, st in enumerate(states):
            r = _predict_residual(st, ecef, clk, isb)
            if abs(r) < gates[j]:
                inliers.append(j)
                cost += r * r
        return inliers, cost

    it = 0
    while it < max_iter:
        if exhaustive:
            idx = candidates[it]
        else:
            idx = tuple(rng.sample(range(m), s))
        it += 1
        iters_done = it
        ev = evaluate(idx)
        if ev is None:
            continue
        inliers, cost = ev
        if len(inliers) < min_consensus:
            continue
        if (len(inliers) > len(best_inliers) or
                (len(inliers) == len(best_inliers) and cost < best_cost)):
            best_inliers, best_cost = inliers, cost
            if not exhaustive:
                w_est = len(inliers) / m
                max_iter = min(max_iter, ransac_iterations(p, w_est, s))

    if not best_inliers:
        return RansacResult(ok=False, fix=None, iterations=iters_done,
                            reason="no consensus set reached min_consensus")

    inlier_ids = [states[j].sat_id for j in best_inliers]
    inlier_set = set(inlier_ids)

    final_states = [st for st in states if st.sat_id in inlier_set]
    final_fix = wls_position(final_states, prior=prior, prior_clock_m=prior_clock_m)
    if final_fix is None:
        return RansacResult(ok=False, fix=None, iterations=iters_done,
                            reason="final WLS over inliers failed")

    ecef = final_fix["ecef"]
    clk = final_fix["clock_bias_m"]
    isb = final_fix.get("isb", {})
    residuals: Dict[str, float] = {}
    thresholds: Dict[str, float] = {}
    for j, st in enumerate(states):
        residuals[st.sat_id] = _predict_residual(st, ecef, clk, isb)
        thresholds[st.sat_id] = gates[j]

    excluded_ids = [st.sat_id for st in states if st.sat_id not in inlier_set]

    result = RansacResult(
        ok=True, fix=final_fix,
        inlier_ids=inlier_ids, excluded_ids=excluded_ids,
        residuals=residuals, thresholds=thresholds,
        consensus_size=len(best_inliers), iterations=iters_done,
        reason="ok",
    )

    if position_prior is not None and gate_m is not None:
        dx = ecef[0] - position_prior[0]
        dy = ecef[1] - position_prior[1]
        dz = ecef[2] - position_prior[2]
        jump = math.sqrt(dx * dx + dy * dy + dz * dz)
        if jump > gate_m:
            result.prior_gate_rejected = True
            result.reason = (f"consensus position jumped {jump:.1f} m > gate {gate_m:.1f} m "
                             f"from prior — possible majority spoof")
    return result
