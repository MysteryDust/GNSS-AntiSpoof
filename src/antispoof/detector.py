"""Fuse the geometric (RANSAC/RAIM) and physical (AGC/C-N0) detectors.

No single indicator is decisive on its own (RANSAC can be fooled by a majority
spoof; RAIM can mislabel the minority; AGC/C-N0 only flags *overpowered*
attacks). Fused, they are strong:

  * **RANSAC** isolates the largest self-consistent satellite subset and names
    the excluded (spoofed) PRNs, and produces the mitigated fix from the inliers.
  * **RAIM** provides the formal integrity test: a fault flag on the full set and
    a self-consistency check on the RANSAC inlier set.
  * **Physical RPM** corroborates with the RF power signature (AGC drop +
    abnormal/uniform C/N0), independent of pseudorange geometry.
  * A **position-prior gate** (inside RANSAC) catches the majority-spoof case
    where the consensus itself is the spoof — the engine then falls back to the
    last trusted position / dead reckoning and raises a hard alarm.

The output is a single :class:`SpoofReport` per epoch: a status, the named
spoofed PRNs, the mitigated ("trusted") fix, the naive all-satellite fix for
comparison, and the contributing reasons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ..core.solver import SatState, wls_position
from .physical import PhysicalReport, RfVerdict
from .ransac import RansacResult, ransac_pvt
from .raim import RaimResult, raim_consistency, raim_fde


class SpoofStatus(str, Enum):
    CLEAN = "CLEAN"
    SUSPECT = "SUSPECT"
    SPOOFED = "SPOOFED"


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class SpoofReport:
    status: SpoofStatus
    confidence: Confidence
    spoofed_prns: List[str] = field(default_factory=list)      # confidently identified culprits
    excluded_prns: List[str] = field(default_factory=list)     # full set removed for mitigation
    trusted_fix: Optional[Dict] = None        # inlier fix
    mitigated: bool = True                     # True if trusted_fix is a trustworthy clean-subset fix
    naive_fix: Optional[Dict] = None          # all-satellite fix (for comparison)
    reasons: List[str] = field(default_factory=list)
    ransac: Optional[RansacResult] = None
    raim: Optional[RaimResult] = None
    physical: Optional[PhysicalReport] = None
    n_sats: int = 0


# Excluded-residual magnitude (m) above which an exclusion is "obviously" a fault.
BIG_RESIDUAL_M = 30.0

# "Coherent spoofed subset" discriminator: the excluded satellites must number at
# least this many, fit their own position to within this RMS, and that position
# must be at least this far from the inlier fix, to be called a spoofed subset
# (vs scattered multipath outliers, which do not form a coherent second fix).
COHERENT_MIN_SATS = 5
COHERENT_RMS_M = 25.0
COHERENT_SEP_M = 100.0


class SpoofDetector:
    def __init__(self, gate_m: float = 80.0, sigma_rho: float = 5.0):
        self.gate_m = gate_m
        self.sigma_rho = sigma_rho

    def process(
        self,
        states: List[SatState],
        physical: Optional[PhysicalReport] = None,
        position_prior=None,
        gate_m: Optional[float] = None,
    ) -> SpoofReport:
        n = len(states)
        naive_fix = wls_position(states, prior=position_prior or _prior_from(states))

        ransac = ransac_pvt(
            states,
            prior=position_prior or _prior_from(states),
            position_prior=position_prior,
            gate_m=(gate_m or self.gate_m) if position_prior is not None else None,
            sigma_rho=self.sigma_rho,
        )
        raim_full = raim_fde(states, sigma_rho=self.sigma_rho)

        reasons: List[str] = []

        if not ransac.ok or ransac.fix is None:
            # Could not form a consensus at all -> treat as suspect/insufficient.
            reasons.append(f"RANSAC failed: {ransac.reason}")
            status = SpoofStatus.SUSPECT if n >= 5 else SpoofStatus.CLEAN
            return SpoofReport(
                status=status, confidence=Confidence.LOW, naive_fix=naive_fix,
                reasons=reasons, ransac=ransac, raim=raim_full, physical=physical, n_sats=n,
            )

        excluded = list(ransac.excluded_ids)
        inlier_states = [s for s in states if s.sat_id in set(ransac.inlier_ids)]
        inlier_consistent, sse_in, T_in, dof_in = raim_consistency(inlier_states, self.sigma_rho)

        max_excl_resid = max((abs(ransac.residuals.get(i, 0.0)) for i in excluded), default=0.0)
        big = max_excl_resid > BIG_RESIDUAL_M
        # Name only confidently-faulted satellites (residual well beyond the gate),
        # so a marginal authentic exclusion is not blamed as a spoofer.
        confident = [i for i in excluded
                     if abs(ransac.residuals.get(i, 0.0)) > max(2.0 * ransac.thresholds.get(i, 20.0),
                                                                BIG_RESIDUAL_M)]
        physical_spoof = physical is not None and physical.verdict == RfVerdict.SPOOFING
        physical_jam = physical is not None and physical.verdict == RfVerdict.JAMMING

        # A self-consistent inlier set with some satellites excluded. On real data
        # this happens routinely (multipath/NLOS), so on its own it is NOT evidence
        # of spoofing — RANSAC simply rejected outliers and the fix is still good.
        geometric_subset = bool(excluded) and inlier_consistent

        # Spoof-specific discriminator: do the EXCLUDED satellites themselves form
        # a *second* internally-consistent fix at a clearly different location? A
        # single-antenna spoofer makes its fake subset mutually consistent with one
        # false position; scattered multipath outliers do not. This separates
        # spoofing from multipath without relying on AGC.
        coherent_spoof_subset = self._excluded_form_coherent_fix(states, excluded, ransac.fix)

        spoof_corroborated = physical_spoof or coherent_spoof_subset

        # ---- decision ----
        if ransac.prior_gate_rejected:
            # The consensus jumped far from the propagated prior. This is either a
            # majority spoof or a severe multipath / bad-geometry epoch. We dead
            # reckon either way (the fix is untrustworthy), but only *call it
            # spoofing* when corroborated by an RF signature or a coherent excluded
            # subset — otherwise it is flagged SUSPECT to avoid mislabelling
            # multipath as an attack on live data.
            reasons.append(ransac.reason)
            if physical is not None:
                reasons.extend(physical.reasons)
            corroborated = physical_spoof or coherent_spoof_subset
            return SpoofReport(
                status=SpoofStatus.SPOOFED if corroborated else SpoofStatus.SUSPECT,
                confidence=Confidence.HIGH if corroborated else Confidence.LOW,
                spoofed_prns=[], excluded_prns=excluded,
                trusted_fix=None, mitigated=False, naive_fix=naive_fix,
                reasons=reasons + [("corroborated majority spoof" if corroborated
                                    else "uncorroborated consensus jump (possible majority spoof or "
                                    "severe multipath)") + "; dead-reckoning from last trusted fix"],
                ransac=ransac, raim=raim_full, physical=physical, n_sats=n,
            )

        if geometric_subset and spoof_corroborated:
            # Geometric subset isolation corroborated by an independent spoof cue
            # (RF power signature and/or a coherent excluded subset). This is a
            # genuine spoofing detection — exclude the subset and keep the fix.
            reasons.append(f"RANSAC isolated a consistent inlier set of {ransac.consensus_size}; "
                           f"excluded {excluded}")
            if coherent_spoof_subset:
                reasons.append("excluded satellites form their own consistent fix at a different "
                               "location — coherent spoofed subset")
            if physical_spoof and physical is not None:
                reasons.extend(physical.reasons)
            conf = Confidence.HIGH if (physical_spoof and coherent_spoof_subset) else (
                Confidence.HIGH if physical_spoof else Confidence.MEDIUM)
            named = confident or (excluded if coherent_spoof_subset else [])
            return SpoofReport(
                status=SpoofStatus.SPOOFED, confidence=conf,
                spoofed_prns=named, excluded_prns=excluded,
                trusted_fix=ransac.fix, mitigated=True, naive_fix=naive_fix,
                reasons=reasons, ransac=ransac, raim=raim_full, physical=physical, n_sats=n,
            )

        if physical_spoof or physical_jam:
            # An RF anomaly with no separable spoofed subset. This is unreliable on
            # its own (environmental RF, jamming, or early small-offset spoof), so
            # we flag SUSPECT but keep trusting the RANSAC fix rather than dead
            # reckoning — geometry shows no coherent takeover yet.
            reasons.extend(physical.reasons if physical else [])
            reasons.append("RF anomaly without a coherent spoofed subset — flagging but trusting "
                           "the outlier-rejected fix")
            return SpoofReport(
                status=SpoofStatus.SUSPECT,
                confidence=Confidence.MEDIUM if physical_spoof else Confidence.LOW,
                spoofed_prns=[], excluded_prns=excluded,
                trusted_fix=ransac.fix, mitigated=True, naive_fix=naive_fix,
                reasons=reasons, ransac=ransac, raim=raim_full, physical=physical, n_sats=n,
            )

        if geometric_subset and big:
            # Outliers excluded with a large residual but no spoof corroboration —
            # most likely multipath/NLOS. Flag SUSPECT, keep the fix.
            reasons.append(f"excluded {excluded} (residual up to {max_excl_resid:.1f} m); "
                           f"no RF or coherent-subset corroboration — likely multipath")
            return SpoofReport(
                status=SpoofStatus.SUSPECT, confidence=Confidence.LOW,
                spoofed_prns=[], excluded_prns=excluded,
                trusted_fix=ransac.fix, mitigated=True, naive_fix=naive_fix,
                reasons=reasons, ransac=ransac, raim=raim_full, physical=physical, n_sats=n,
            )

        if excluded or raim_full.fault_detected:
            # Routine small outlier rejection — normal navigation, position trusted.
            reasons.append("routine outlier rejection (RANSAC/RAIM); no spoofing indicators")
            return SpoofReport(
                status=SpoofStatus.CLEAN, confidence=Confidence.MEDIUM,
                spoofed_prns=[], excluded_prns=excluded,
                trusted_fix=ransac.fix, mitigated=True, naive_fix=naive_fix,
                reasons=reasons, ransac=ransac, raim=raim_full, physical=physical, n_sats=n,
            )

        reasons.append("all consistency checks passed")
        return SpoofReport(
            status=SpoofStatus.CLEAN, confidence=Confidence.HIGH,
            spoofed_prns=[], excluded_prns=excluded,
            trusted_fix=ransac.fix, mitigated=True, naive_fix=naive_fix,
            reasons=reasons, ransac=ransac, raim=raim_full, physical=physical, n_sats=n,
        )

    def _excluded_form_coherent_fix(self, states, excluded, inlier_fix) -> bool:
        """True if the excluded satellites form their own consistent fix far from
        the inlier fix (signature of a single-antenna spoofed subset, vs scattered
        multipath which does not form a coherent second solution)."""
        if inlier_fix is None or len(excluded) < COHERENT_MIN_SATS:
            return False
        excl_states = [s for s in states if s.sat_id in set(excluded)]
        fix = wls_position(excl_states, prior=inlier_fix["ecef"], compute_dop_flag=False)
        if fix is None or fix.get("rms_residual_m", 1e9) > COHERENT_RMS_M:
            return False
        a, b = fix["ecef"], inlier_fix["ecef"]
        sep = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
        return sep > COHERENT_SEP_M


def _prior_from(states: List[SatState]):
    if not states:
        return (4438000.0, 3086000.0, 3375000.0)
    s = states[0]
    import math
    norm = math.sqrt(s.x ** 2 + s.y ** 2 + s.z ** 2)
    return (s.x / norm * 6371000.0, s.y / norm * 6371000.0, s.z / norm * 6371000.0)
