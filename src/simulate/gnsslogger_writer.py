"""Write :class:`EpochObs` streams as Android GnssLogger 'Raw' CSV.

This closes the loop: the simulator produces ``EpochObs`` with true (or spoofed)
pseudoranges, this writer back-encodes them into the exact GnssLogger raw fields
(TimeNanos / FullBiasNanos / ReceivedSvTimeNanos / TimeOffsetNanos / ...), and
``src.io.gnsslogger`` parses them straight back. So the *real-data path* is
exercised end-to-end by synthetic data, and a real phone log flows through the
identical code.

Encoding choices (all self-consistent with the parser):
  * BiasNanos = 0, TimeOffsetNanos = 0  -> the two TimeOffsetNanos conventions
    coincide and there is no ambiguity.
  * A fixed hardware-clock base T0 and a constant FullBiasNanos are chosen so
    that  TimeNanos - FullBiasNanos == absolute GPS nanoseconds  for every epoch.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional

from ..core.measurements import EpochObs, RawMeasurement
from ..core.solver import carrier_freq_for
from ..core.timeutils import datetime_to_gps_sow

SPEED_OF_LIGHT = 2.99792458e8
NS_PER_WEEK = 604800 * 1_000_000_000
BDS_GPS_OFFSET_NS = 14 * 1_000_000_000
T0_UPTIME_NS = 1_000_000_000_000  # arbitrary device-uptime base for TimeNanos

SYS_TO_CONSTELLATION = {"G": 1, "S": 2, "R": 3, "J": 4, "C": 5, "E": 6, "I": 7}

HEADER_COLUMNS = [
    "utcTimeMillis", "TimeNanos", "LeapSecond", "TimeUncertaintyNanos",
    "FullBiasNanos", "BiasNanos", "BiasUncertaintyNanos", "DriftNanosPerSecond",
    "DriftUncertaintyNanosPerSecond", "HardwareClockDiscontinuityCount", "Svid",
    "TimeOffsetNanos", "State", "ReceivedSvTimeNanos", "ReceivedSvTimeUncertaintyNanos",
    "Cn0DbHz", "PseudorangeRateMetersPerSecond", "PseudorangeRateUncertaintyMetersPerSecond",
    "AccumulatedDeltaRangeState", "AccumulatedDeltaRangeMeters",
    "AccumulatedDeltaRangeUncertaintyMeters", "CarrierFrequencyHz", "CarrierCycles",
    "CarrierPhase", "CarrierPhaseUncertainty", "MultipathIndicator", "SnrInDb",
    "ConstellationType", "AgcDb",
]

STATE_VALID = 0x1 | 0x8  # CODE_LOCK | TOW_DECODED
LEAP_SECONDS = 18


def _band_for(carrier_hz: float) -> str:
    mhz = carrier_hz / 1e6
    if 1565 <= mhz <= 1612:
        return "L1"
    if 1160 <= mhz <= 1300:
        return "L5"
    return f"{round(mhz)}MHz"


def write_gnsslogger_csv(
    epochs: List[EpochObs],
    path: str,
    header_comment: str = "Synthetic GnssLogger log (GNSS-AntiSpoof simulator)",
) -> None:
    if not epochs:
        raise ValueError("no epochs to write")

    gps0 = datetime_to_gps_sow(epochs[0].time, leap_seconds=0)
    full_bias_ns = round(T0_UPTIME_NS - gps0 * 1e9)  # constant, negative

    lines: List[str] = []
    lines.append(f"# {header_comment}")
    lines.append("# Header Description:")
    lines.append("Raw," + ",".join(HEADER_COLUMNS))

    for ep in epochs:
        gps_s = datetime_to_gps_sow(ep.time, leap_seconds=0)
        time_nanos = round(T0_UPTIME_NS + (gps_s - gps0) * 1e9)
        t_rx_gnss = time_nanos - full_bias_ns  # == gps_s * 1e9
        utc_ms = int(round((gps_s - LEAP_SECONDS) * 1000.0)) + int(round(_gps_epoch_unix_ms()))

        for m in ep.sats:
            carrier = m.carrier_freq_hz or carrier_freq_for(m.sat_id)
            # Receive time in the constellation's frame.
            if m.sys == "C":
                t_rx = (t_rx_gnss % NS_PER_WEEK) - BDS_GPS_OFFSET_NS
            else:
                t_rx = t_rx_gnss % NS_PER_WEEK
            pr_ns = m.pseudorange_m / SPEED_OF_LIGHT * 1e9
            recv_sv_time = round(t_rx - pr_ns)  # TimeOffsetNanos = 0
            sv_unc = ""
            if m.pr_uncertainty_m:
                sv_unc = f"{m.pr_uncertainty_m / SPEED_OF_LIGHT * 1e9:.1f}"
            band = _band_for(carrier)
            agc = ep.agc_db.get(band)
            if agc is None and ep.agc_db:
                agc = next(iter(ep.agc_db.values()))
            ctype = SYS_TO_CONSTELLATION.get(m.sys, 0)

            rowmap = {
                "utcTimeMillis": utc_ms,
                "TimeNanos": time_nanos,
                "LeapSecond": LEAP_SECONDS,
                "FullBiasNanos": full_bias_ns,
                "BiasNanos": 0,
                "HardwareClockDiscontinuityCount": 0,
                "Svid": m.prn,
                "TimeOffsetNanos": 0,
                "State": STATE_VALID,
                "ReceivedSvTimeNanos": recv_sv_time,
                "ReceivedSvTimeUncertaintyNanos": sv_unc,
                "Cn0DbHz": f"{m.cn0_dbhz:.1f}" if m.cn0_dbhz is not None else "",
                "PseudorangeRateMetersPerSecond": f"{m.pr_rate_mps:.4f}" if m.pr_rate_mps is not None else "",
                "CarrierFrequencyHz": f"{carrier:.0f}",
                "ConstellationType": ctype,
                "AgcDb": f"{agc:.2f}" if agc is not None else "",
            }
            row = [str(rowmap.get(col, "")) for col in HEADER_COLUMNS]
            lines.append("Raw," + ",".join(row))

    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")


def _gps_epoch_unix_ms() -> float:
    # GPS epoch 1980-01-06 in Unix ms (constant); avoids importing datetime maths inline.
    # 1980-01-06T00:00:00Z = 315964800 s since Unix epoch.
    return 315964800.0 * 1000.0
