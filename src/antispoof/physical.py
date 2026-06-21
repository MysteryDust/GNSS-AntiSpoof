"""Physical-layer spoofing/jamming detection from AGC and C/N0.

GNSS signals arrive far below the thermal-noise floor (~ -130 dBm). To capture a
receiver's tracking loops a spoofer must inject *extra* RF power, which leaves
two measurable fingerprints exposed by the Android raw API:

  * **AGC (Automatic Gain Control)** — the front-end lowers its gain when total
    in-band power rises, so an overpowered attack shows up as a sudden, sustained
    AGC *drop* from the open-sky baseline (Akos 2012).
  * **C/N0 (carrier-to-noise density)** — an overpowered spoofer drives C/N0
    abnormally high and unnaturally *uniform* across satellites (a single
    transmit antenna gives every "satellite" the same power), unlike genuine
    signals whose C/N0 spreads with elevation/multipath.

Receiver Power Monitoring (RPM) fuses the two (Miralles 2018 / Spens 2022):

    AGC        | C/N0                  | verdict
    -----------+-----------------------+----------------------------
    unchanged  | normal                | NOMINAL
    drop       | drops proportionally  | JAMMING (broadband RFI)
    drop       | stable / high         | SPOOFING (overpowered)
    unchanged  | abnormally high+uniform| SPOOFING (single antenna)

AGC polarity and absolute level are device-specific, so we learn an open-sky
*baseline* from the first clean epochs and detect *step changes* (k-sigma), not
absolute thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional
from collections import deque

from ..core.measurements import EpochObs


class RfVerdict(str, Enum):
    NOMINAL = "NOMINAL"
    JAMMING = "JAMMING"
    SPOOFING = "SPOOFING"
    UNKNOWN = "UNKNOWN"      # not enough info (e.g. no AGC reported, baseline not learned)


@dataclass
class PhysicalReport:
    verdict: RfVerdict = RfVerdict.UNKNOWN
    agc_db: Optional[float] = None
    agc_baseline: Optional[float] = None
    agc_drop_db: Optional[float] = None        # baseline - current (positive = power added)
    agc_anomaly: bool = False
    cn0_median: Optional[float] = None
    cn0_std: Optional[float] = None
    cn0_high: bool = False                     # median > high threshold
    cn0_uniform: bool = False                  # cross-SV std < uniform threshold
    cn0_dropped: bool = False
    reasons: List[str] = field(default_factory=list)


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _std(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return 0.0 if vals else None
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


class PhysicalMonitor:
    """Stateful AGC/C/N0 monitor. Feed it epochs in time order.

    The first ``baseline_epochs`` epochs are assumed nominal (open sky) and used
    to learn the AGC baseline mean/std and a nominal C/N0 median. Thresholds are
    deliberately conservative defaults from the literature and are all tunable.
    """

    def __init__(
        self,
        baseline_epochs: int = 8,
        agc_k_sigma: float = 4.0,
        agc_min_step_db: float = 3.0,
        cn0_high_dbhz: float = 50.0,
        cn0_uniform_std_dbhz: float = 2.0,
        cn0_drop_dbhz: float = 6.0,
        window: int = 30,
    ):
        self.baseline_epochs = baseline_epochs
        self.agc_k_sigma = agc_k_sigma
        self.agc_min_step_db = agc_min_step_db
        self.cn0_high = cn0_high_dbhz
        self.cn0_uniform_std = cn0_uniform_std_dbhz
        self.cn0_drop = cn0_drop_dbhz
        self._agc_hist: Deque[float] = deque(maxlen=window)
        self._cn0_base_hist: List[float] = []
        self._agc_baseline: Optional[float] = None
        self._agc_baseline_std: float = 0.5
        self._cn0_baseline_median: Optional[float] = None
        self._n_seen = 0
        # An overpowered attack produces a *sustained* AGC drop; real environments
        # produce transient dips. Require the AGC anomaly to persist before calling
        # it spoofing/jamming, to cut false alarms on live data.
        self._anom_streak = 0
        self.sustain_epochs = 3

    @property
    def baseline_ready(self) -> bool:
        return self._agc_baseline is not None or self._cn0_baseline_median is not None

    def _learn_baseline(self, agc: Optional[float], cn0_median: Optional[float]):
        if agc is not None:
            self._agc_hist.append(agc)
        if cn0_median is not None:
            self._cn0_base_hist.append(cn0_median)
        if self._n_seen >= self.baseline_epochs:
            if self._agc_hist:
                vals = list(self._agc_hist)
                self._agc_baseline = sum(vals) / len(vals)
                self._agc_baseline_std = max(_std(vals) or 0.5, 0.2)
            if self._cn0_base_hist:
                self._cn0_baseline_median = _median(self._cn0_base_hist)

    def update(self, epoch: EpochObs) -> PhysicalReport:
        self._n_seen += 1
        agc = epoch.mean_agc()
        cn0_vals = [m.cn0_dbhz for m in epoch.sats if m.cn0_dbhz is not None]
        cn0_median = _median(cn0_vals)
        cn0_std = _std(cn0_vals)

        rep = PhysicalReport(agc_db=agc, cn0_median=cn0_median, cn0_std=cn0_std)

        # Learn baseline during the initial nominal window.
        if not self.baseline_ready or self._n_seen <= self.baseline_epochs:
            self._learn_baseline(agc, cn0_median)
            rep.verdict = RfVerdict.UNKNOWN
            rep.agc_baseline = self._agc_baseline
            rep.reasons.append("learning baseline")
            return rep

        rep.agc_baseline = self._agc_baseline

        # --- AGC step detection (power added if AGC drops below baseline) ---
        agc_drop = None
        if agc is not None and self._agc_baseline is not None:
            agc_drop = self._agc_baseline - agc
            rep.agc_drop_db = agc_drop
            step_thresh = max(self.agc_k_sigma * self._agc_baseline_std, self.agc_min_step_db)
            if agc_drop > step_thresh:
                rep.agc_anomaly = True
                rep.reasons.append(
                    f"AGC dropped {agc_drop:.1f} dB > {step_thresh:.1f} dB threshold (added RF power)")

        # --- C/N0 anomalies ---
        if cn0_median is not None:
            if cn0_median > self.cn0_high:
                rep.cn0_high = True
                rep.reasons.append(f"median C/N0 {cn0_median:.1f} > {self.cn0_high} dB-Hz (overpowered)")
            if self._cn0_baseline_median is not None and \
                    cn0_median < self._cn0_baseline_median - self.cn0_drop:
                rep.cn0_dropped = True
                rep.reasons.append(
                    f"median C/N0 fell {self._cn0_baseline_median - cn0_median:.1f} dB (signal loss)")
        if cn0_std is not None and len(cn0_vals) >= 5 and cn0_std < self.cn0_uniform_std:
            rep.cn0_uniform = True
            rep.reasons.append(
                f"C/N0 spread {cn0_std:.1f} < {self.cn0_uniform_std} dB-Hz across {len(cn0_vals)} SVs (single antenna)")

        # Sustained-anomaly tracking for AGC-based verdicts.
        self._anom_streak = self._anom_streak + 1 if rep.agc_anomaly else 0

        # --- RPM truth table ---
        rep.verdict = self._rpm(rep)
        return rep

    def _rpm(self, rep: PhysicalReport) -> RfVerdict:
        sustained = self._anom_streak >= self.sustain_epochs
        if rep.agc_anomaly and sustained:
            # Sustained added RF power detected.
            if rep.cn0_dropped:
                return RfVerdict.JAMMING          # power up, signal down -> broadband RFI
            return RfVerdict.SPOOFING             # power up, C/N0 stable/high -> overpowered spoof
        # No (sustained) AGC step: fall back to spoof-specific C/N0 shape cues.
        if rep.cn0_high and rep.cn0_uniform:
            return RfVerdict.SPOOFING             # matched-power single-antenna spoof
        if rep.cn0_uniform and rep.cn0_median is not None and rep.cn0_median > 45.0:
            return RfVerdict.SPOOFING
        return RfVerdict.NOMINAL
