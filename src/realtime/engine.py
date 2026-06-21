"""Real-time standalone-positioning + anti-spoofing engine.

Feed it epochs one at a time, in order, as they would arrive from a phone's
``GnssMeasurementsEvent`` callback (or from a streamed log). Per epoch it:

  1. builds satellite states from broadcast ephemeris (standalone — it never
     consults the OS Fused Location Provider, so OS-level / network spoofing is
     bypassed by construction);
  2. updates the physical-layer AGC/C-N0 monitor;
  3. runs the fused spoof detector (RANSAC + RAIM + RPM) using the *last trusted
     position* as a prior gate;
  4. emits a trusted (mitigated) PVT plus a full spoofing report.

State carried between epochs: the last trusted ECEF fix (prior gate + solver
seed) and a short dead-reckoning fallback when the current epoch is judged
untrustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from ..antispoof.detector import SpoofDetector, SpoofReport, SpoofStatus
from ..antispoof.physical import PhysicalMonitor, RfVerdict
from ..core.coordinates import ecef_to_geodetic
from ..core.measurements import EpochObs
from ..core.solver import compute_sat_states, wls_velocity
from ..core.timeutils import SECONDS_PER_WEEK, datetime_to_gps_sow

DEFAULT_PRIOR_XYZ = (4438000.0, 3086000.0, 3375000.0)


@dataclass
class EngineOutput:
    time: datetime                                  # GPS time
    utc_time: datetime                              # leap-corrected UTC
    status: SpoofStatus
    confidence: str
    n_sats: int
    spoofed_prns: List[str] = field(default_factory=list)
    # trusted / mitigated solution
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_m: Optional[float] = None
    ecef: Optional[tuple] = None
    speed_mps: Optional[float] = None
    vx_mps: Optional[float] = None
    vy_mps: Optional[float] = None
    vz_mps: Optional[float] = None
    clock_bias_m: Optional[float] = None
    pdop: Optional[float] = None
    hdop: Optional[float] = None
    # naive all-satellite solution (what an unprotected receiver would report)
    naive_lat_deg: Optional[float] = None
    naive_lon_deg: Optional[float] = None
    naive_alt_m: Optional[float] = None
    # diagnostics
    rf_verdict: Optional[str] = None
    agc_drop_db: Optional[float] = None
    raim_sse: Optional[float] = None
    raim_threshold: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    dead_reckoned: bool = False
    report: Optional[SpoofReport] = None


class RealtimeEngine:
    def __init__(
        self,
        nav_data,
        glonass_slot_to_k: Optional[Dict[int, int]] = None,
        leap_seconds: int = 18,
        elevation_mask_deg: float = 10.0,
        gate_m: float = 80.0,
        sigma_rho: float = 5.0,
        physical_baseline_epochs: int = 8,
        initial_prior=DEFAULT_PRIOR_XYZ,
    ):
        self.nav_data = nav_data
        self.slots = glonass_slot_to_k or {}
        self.leap_seconds = leap_seconds
        self.elevation_mask_rad = math.radians(elevation_mask_deg)
        self.detector = SpoofDetector(gate_m=gate_m, sigma_rho=sigma_rho)
        self.physical = PhysicalMonitor(baseline_epochs=physical_baseline_epochs)
        # Propagated prior state: position + velocity + time. The prior is
        # advanced *every* epoch (by the trusted fix, or by dead-reckoning when
        # the epoch is untrusted) so the gate always compares against a current
        # prediction rather than a stale fix.
        self._prior_pos = None
        self._prior_vel = None
        self._prior_time: Optional[datetime] = None
        self._last_trusted_time: Optional[datetime] = None  # time of last genuinely trusted fix
        self._have_trusted = False
        self._seed = initial_prior
        self.gate_m = gate_m

    # Gate behaviour during dead reckoning: the prior gate widens with the time
    # since the last trusted fix (dead-reckoning uncertainty grows), and after a
    # sustained outage it disengages entirely so the engine can re-acquire rather
    # than locking onto a stale prior forever.
    GATE_GROWTH_MPS = 25.0
    GATE_MAX_M = 2000.0
    REACQUIRE_AFTER_S = 30.0

    def _predict_prior(self, epoch_time: datetime):
        """One-step prediction of the receiver position to this epoch's time."""
        if self._prior_pos is None:
            return None, None
        if self._prior_time is None or self._prior_vel is None:
            return self._prior_pos, None
        dt = (epoch_time - self._prior_time).total_seconds()
        if not (0 < dt < 30):
            return self._prior_pos, dt
        vx, vy, vz = self._prior_vel
        pred = (self._prior_pos[0] + vx * dt,
                self._prior_pos[1] + vy * dt,
                self._prior_pos[2] + vz * dt)
        return pred, dt

    def process(self, epoch: EpochObs) -> EngineOutput:
        phys = self.physical.update(epoch)
        predicted_prior, _dt = self._predict_prior(epoch.time)
        seed = predicted_prior or self._prior_pos or self._seed

        states = compute_sat_states(
            epoch, self.nav_data, self.slots,
            rx_prior=seed,
            elevation_mask_rad=self.elevation_mask_rad,
        )

        utc_time = (epoch.time - timedelta(seconds=self.leap_seconds)).astimezone(timezone.utc)
        out = EngineOutput(
            time=epoch.time, utc_time=utc_time,
            status=SpoofStatus.CLEAN, confidence="LOW", n_sats=len(states),
            rf_verdict=phys.verdict.value, agc_drop_db=phys.agc_drop_db,
        )

        if len(states) < 5:
            out.reasons = [f"only {len(states)} usable satellites — cannot solve/monitor"]
            return self._dead_reckon(out, epoch, predicted_prior)

        # The prior gate is only enabled once we have established a trusted fix.
        # While the RF environment looks clean, the gate widens with dead-reckoning
        # age and finally disengages after a sustained *benign* outage, so a genuine
        # signal loss / drift can re-acquire instead of locking forever. But while
        # the physical layer is actively flagging an attack we keep the gate tight
        # and engaged — re-acquiring during a detected spoof would walk onto it.
        gate_prior = None
        gate_m = self.gate_m
        attack_ongoing = phys.verdict in (RfVerdict.SPOOFING, RfVerdict.JAMMING)
        if self._have_trusted and self._last_trusted_time is not None:
            dr_age = (epoch.time - self._last_trusted_time).total_seconds()
            if attack_ongoing:
                gate_prior = predicted_prior          # tight gate, stay protected
                gate_m = self.gate_m
            elif dr_age <= self.REACQUIRE_AFTER_S:
                gate_prior = predicted_prior          # benign outage: widen with DR uncertainty
                gate_m = min(self.gate_m + self.GATE_GROWTH_MPS * max(dr_age, 0.0), self.GATE_MAX_M)
            else:
                self._prior_vel = None                # benign sustained outage: re-acquire
        report = self.detector.process(states, physical=phys,
                                       position_prior=gate_prior, gate_m=gate_m)
        out.report = report
        out.status = report.status
        out.confidence = report.confidence.value
        out.spoofed_prns = report.spoofed_prns
        out.reasons = report.reasons
        if report.raim is not None:
            out.raim_sse = report.raim.sse
            out.raim_threshold = report.raim.threshold

        if report.naive_fix is not None:
            nlat, nlon, nalt = ecef_to_geodetic(*report.naive_fix["ecef"])
            out.naive_lat_deg = math.degrees(nlat)
            out.naive_lon_deg = math.degrees(nlon)
            out.naive_alt_m = nalt

        trusted = report.trusted_fix
        if trusted is None or not report.mitigated:
            # Untrustworthy epoch: either a majority spoof (no fix) or a physical-
            # only flag where we could not certify a clean satellite subset. Dead
            # reckon and keep advancing the prior by prediction so it tracks
            # expected motion instead of drifting along with the spoof.
            return self._dead_reckon(out, epoch, predicted_prior)

        ecef = trusted["ecef"]
        lat, lon, alt = ecef_to_geodetic(*ecef)
        out.lat_deg = math.degrees(lat)
        out.lon_deg = math.degrees(lon)
        out.alt_m = alt
        out.ecef = ecef
        out.clock_bias_m = trusted.get("clock_bias_m")
        dop = trusted.get("dop", {})
        out.pdop = dop.get("pdop")
        out.hdop = dop.get("hdop")

        # Velocity from the trusted satellites' Doppler.
        inlier_states = [s for s in states if s.sat_id in trusted["satellites"]]
        vel = wls_velocity(inlier_states, ecef)
        if vel is not None:
            out.vx_mps, out.vy_mps, out.vz_mps = vel["vx"], vel["vy"], vel["vz"]
            out.speed_mps = math.sqrt(vel["vx"] ** 2 + vel["vy"] ** 2 + vel["vz"] ** 2)

        # Advance the propagated prior with the trusted fix. Clear a stale velocity
        # if this fix produced no velocity solution, so we never extrapolate the
        # next prior with an out-of-date velocity.
        self._prior_pos = ecef
        self._prior_vel = (vel["vx"], vel["vy"], vel["vz"]) if vel is not None else None
        self._prior_time = epoch.time
        self._last_trusted_time = epoch.time
        self._have_trusted = True
        return out

    def _dead_reckon(self, out: EngineOutput, epoch: EpochObs, predicted_prior) -> EngineOutput:
        """Propagate position by the last velocity when the epoch is untrusted."""
        if predicted_prior is None:
            return out
        ecef = predicted_prior
        lat, lon, alt = ecef_to_geodetic(*ecef)
        out.lat_deg = math.degrees(lat)
        out.lon_deg = math.degrees(lon)
        out.alt_m = alt
        out.ecef = ecef
        out.dead_reckoned = True
        out.reasons = out.reasons + ["dead-reckoned from last trusted fix (prior propagated by velocity)"]
        # Keep the prior moving so it tracks expected motion and does not go stale.
        self._prior_pos = ecef
        self._prior_time = epoch.time
        return out

    def run(self, epochs: List[EpochObs]) -> List[EngineOutput]:
        return [self.process(ep) for ep in epochs]
