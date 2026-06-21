"""Unified, source-agnostic GNSS measurement model.

Both the RINEX observation parser (``src.io.rinex_stream``) and the Android
GnssLogger parser (``src.io.gnsslogger``) normalise their raw input into the
data structures defined here. Everything downstream — the WLS solver, the
RANSAC / RAIM anti-spoofing engine and the real-time pipeline — operates only
on this model and therefore does not care where the measurements came from.

A note on time: ``EpochObs.time`` is always **GPS time** expressed as a
timezone-aware ``datetime`` (i.e. UTC clock value without the leap-second
offset removed). The solver works in GPS seconds-of-week throughout; UTC is
only produced at the very end for human-facing output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# Constellation single-letter codes used throughout the project (RINEX style).
SYS_GPS = "G"
SYS_GALILEO = "E"
SYS_BEIDOU = "C"
SYS_GLONASS = "R"
SYS_QZSS = "J"
SYS_SBAS = "S"
SYS_IRNSS = "I"

# Android ``ConstellationType`` integer -> RINEX system letter.
# (android.location.GnssStatus constellation constants.)
CONSTELLATION_TYPE_TO_SYS = {
    1: SYS_GPS,       # CONSTELLATION_GPS
    2: SYS_SBAS,      # CONSTELLATION_SBAS
    3: SYS_GLONASS,   # CONSTELLATION_GLONASS
    4: SYS_QZSS,      # CONSTELLATION_QZSS
    5: SYS_BEIDOU,    # CONSTELLATION_BEIDOU
    6: SYS_GALILEO,   # CONSTELLATION_GALILEO
    7: SYS_IRNSS,     # CONSTELLATION_IRNSS / NavIC
}


@dataclass
class RawMeasurement:
    """One satellite's observation at one epoch, normalised across sources.

    ``pseudorange_m`` is the geometric+clock pseudorange in metres. For RINEX it
    is read directly from the C1C/C2I/... observable; for Android it is
    reconstructed from the raw hardware clock fields (see ``gnsslogger.py``).
    """

    sat_id: str                                   # e.g. "G05", "E11", "C20"
    sys: str                                      # one of SYS_* above
    prn: int
    pseudorange_m: float
    pr_rate_mps: Optional[float] = None           # range rate (m/s); +ve = range increasing
    cn0_dbhz: Optional[float] = None              # carrier-to-noise density
    carrier_freq_hz: Optional[float] = None       # signal carrier frequency
    pr_uncertainty_m: Optional[float] = None       # 1-sigma pseudorange uncertainty (Android)
    code: Optional[str] = None                    # RINEX obs code used, or "ANDROID"

    def __post_init__(self):
        if self.sys is None and self.sat_id:
            self.sys = self.sat_id[0]


@dataclass
class EpochObs:
    """All satellite measurements at one epoch plus receiver-level RF metrics.

    ``agc_db`` maps a constellation/frequency key to the receiver's Automatic
    Gain Control level in dB. A sudden drop here is a primary physical indicator
    of an overpowered spoofing/jamming attack (see ``antispoof.physical``).
    RINEX logs carry no AGC, so the dict is simply empty for that source.
    """

    time: datetime                                          # GPS time, tz-aware
    sats: List[RawMeasurement] = field(default_factory=list)
    agc_db: Dict[str, float] = field(default_factory=dict)  # band key -> AGC level (dB)

    @property
    def n_sats(self) -> int:
        return len(self.sats)

    def by_sys(self) -> Dict[str, List[RawMeasurement]]:
        out: Dict[str, List[RawMeasurement]] = {}
        for m in self.sats:
            out.setdefault(m.sys, []).append(m)
        return out

    def mean_agc(self) -> Optional[float]:
        """Mean AGC across the bands reported this epoch, or None if unavailable."""
        vals = [v for v in self.agc_db.values() if v is not None]
        return sum(vals) / len(vals) if vals else None

    def mean_cn0(self) -> Optional[float]:
        vals = [m.cn0_dbhz for m in self.sats if m.cn0_dbhz is not None]
        return sum(vals) / len(vals) if vals else None
