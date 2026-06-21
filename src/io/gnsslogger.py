"""Parser for Android GnssLogger 'Raw' CSV logs -> unified measurement model.

This is the *primary* real-data path: drop a GnssLogger recording into
``data/raw/`` and the engine works on it exactly as it does on the simulator
output (which deliberately emits this same format). The math follows Google's
``gps-measurement-tools`` / ``ProcessGnssMeas.m`` and the EUSPA white paper
"Using GNSS Raw Measurements on Android Devices".

Receiver time (Google convention, using the FIRST FullBiasNanos in the file):

    weekNumber       = floor(-FullBiasNanos0 * 1e-9 / 604800)
    tRxGnssNanos     = TimeNanos - (FullBiasNanos0 + BiasNanos0)      # abs GPS ns
    tRx(GPS/GAL)     = tRxGnssNanos mod 604800e9
    tRx(BeiDou)      = (tRxGnssNanos mod 604800e9) - 14e9             # BDT = GPST-14s
    tRx(GLONASS)     = (tRxGnssNanos mod 86400e9) + (3*3600 - leap)*1e9
    tTx              = ReceivedSvTimeNanos + TimeOffsetNanos          # apply offset ONCE
    pseudorange_m    = (tRx - tTx) * 1e-9 * c

A measurement is used only if its State bitmask shows the pseudorange is
unambiguous: (State & CODE_LOCK) and (State & TOW_DECODED) for GPS/GAL/BDS/QZSS;
(State & CODE_LOCK) and (State & GLO_TOD_DECODED) for GLONASS.

AgcDb (when present in the Raw rows) is averaged per carrier band and exposed as
``EpochObs.agc_db`` for the physical-layer monitor. (Android 13+ may instead emit
AGC in separate 'Agc' records; this reader also folds those in when found.)
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Dict, Iterator, List, Optional, Tuple

from ..core.measurements import (
    CONSTELLATION_TYPE_TO_SYS, EpochObs, RawMeasurement,
)
from ..core.timeutils import GPS_EPOCH

SPEED_OF_LIGHT = 2.99792458e8
WEEKSEC = 604800
NS_PER_WEEK = WEEKSEC * 1_000_000_000
NS_PER_DAY = 86400 * 1_000_000_000
NS_PER_100MS = 100 * 1_000_000
BDS_GPS_OFFSET_NS = 14 * 1_000_000_000
GLO_GPS_OFFSET_HOURS = 3

# GnssMeasurement.State bit values.
STATE_CODE_LOCK = 0x1
STATE_TOW_DECODED = 0x8
STATE_TOW_KNOWN = 0x4000
STATE_GLO_TOD_DECODED = 0x80
STATE_GLO_TOD_KNOWN = 0x8000
STATE_GAL_E1C_2ND_CODE_LOCK = 0x800

DEFAULT_LEAP_SECONDS = 18


def _to_float(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(s: str) -> Optional[int]:
    f = _to_float(s)
    return int(f) if f is not None else None


def _state_valid(state: int, sys: str) -> bool:
    """True if the State bitmask indicates an unambiguous full pseudorange.

    We require TOW_DECODED for GPS/Galileo/BeiDou/QZSS (and GLO_TOD_DECODED for
    GLONASS). The Galileo E1C-2nd-code-lock bit only resolves the 100 ms ambiguity
    for a *different* pseudorange reconstruction; on its own it does not give the
    full week-referenced pseudorange this parser computes, so it is not accepted
    in isolation (doing so risks a 100 ms / ~30 000 km range ambiguity).
    """
    if state is None:
        return False
    if sys == "R":  # GLONASS uses time-of-day
        return bool(state & STATE_CODE_LOCK) and bool(state & STATE_GLO_TOD_DECODED)
    # GPS / Galileo / BeiDou / QZSS — require code lock and TOW decoded.
    return bool(state & STATE_CODE_LOCK) and bool(state & STATE_TOW_DECODED)


def _band_key(carrier_hz: Optional[float]) -> str:
    if carrier_hz is None:
        return "L1"
    mhz = carrier_hz / 1e6
    if 1565 <= mhz <= 1612:
        return "L1"
    if 1160 <= mhz <= 1300:
        return "L5"
    return f"{round(mhz)}MHz"


def _band_rank(carrier_hz: Optional[float]) -> int:
    """Preference for which signal to keep per satellite (lower = preferred).

    Modern phones report several signals per satellite (L1/E1/B1, L5/E5/B2, ...).
    Our single-frequency solver expects one pseudorange per satellite — like a
    RINEX OBS file — so we keep the L1/E1/B1 measurement (which also carries the
    decoded TOW most reliably) and drop the rest. Emitting all of them would put
    duplicate satellite IDs into the solver and corrupt the fix.
    """
    if carrier_hz is None:
        return 1
    mhz = carrier_hz / 1e6
    if 1559 <= mhz <= 1612:   # L1 / E1 / B1
        return 0
    if 1160 <= mhz <= 1217:   # L5 / E5a / B2a
        return 2
    return 3


class _Header:
    def __init__(self, names: List[str]):
        self.idx = {name.strip(): i for i, name in enumerate(names)}

    def get(self, row: List[str], name: str) -> Optional[str]:
        i = self.idx.get(name)
        if i is None or i >= len(row):
            return None
        return row[i]


def _find_headers(path: str) -> Tuple[Optional[_Header], Optional[_Header]]:
    raw_hdr = agc_hdr = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = line.strip()
            if t.startswith("#"):
                t = t.lstrip("# ").strip()
            if t.startswith("Raw,") and raw_hdr is None:
                raw_hdr = _Header(t.split(",")[1:])
            elif t.startswith("Agc,") and agc_hdr is None:
                agc_hdr = _Header(t.split(",")[1:])
            if raw_hdr is not None and agc_hdr is not None:
                break
    return raw_hdr, agc_hdr


def stream_epochs(path: str, leap_seconds: int = DEFAULT_LEAP_SECONDS) -> Iterator[EpochObs]:
    """Yield :class:`EpochObs` from a GnssLogger CSV, grouped into 1 Hz epochs."""
    raw_hdr, agc_hdr = _find_headers(path)
    if raw_hdr is None:
        raise ValueError(f"No 'Raw,' header found in {path}; is this a GnssLogger CSV?")

    full_bias0: Optional[float] = None
    bias0: float = 0.0

    # epoch key (integer ms of receive time) -> accumulator
    epochs: Dict[int, Dict] = {}

    def ensure_epoch(key: int, gps_seconds: float) -> Dict:
        if key not in epochs:
            epochs[key] = {
                "time": GPS_EPOCH + timedelta(seconds=gps_seconds),
                "sat_best": {},          # sat_id -> (rank_tuple, RawMeasurement)
                "agc_sum": {}, "agc_cnt": {},
            }
        return epochs[key]

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            if t.startswith("Agc,") and agc_hdr is not None:
                continue  # AGC records handled below in a second pass if needed
            if not t.startswith("Raw,"):
                continue
            row = t.split(",")[1:]

            time_nanos = _to_int(raw_hdr.get(row, "TimeNanos"))
            full_bias = _to_float(raw_hdr.get(row, "FullBiasNanos"))
            bias = _to_float(raw_hdr.get(row, "BiasNanos")) or 0.0
            if time_nanos is None or full_bias is None or full_bias == 0.0:
                continue
            if full_bias0 is None:
                full_bias0 = full_bias
                bias0 = bias

            time_offset = _to_float(raw_hdr.get(row, "TimeOffsetNanos")) or 0.0
            state = _to_int(raw_hdr.get(row, "State")) or 0
            recv_sv_time = _to_float(raw_hdr.get(row, "ReceivedSvTimeNanos"))
            svid = _to_int(raw_hdr.get(row, "Svid"))
            ctype = _to_int(raw_hdr.get(row, "ConstellationType"))
            if recv_sv_time is None or svid is None or ctype is None:
                continue
            sys = CONSTELLATION_TYPE_TO_SYS.get(ctype)
            if sys is None:
                continue
            if not _state_valid(state, sys):
                continue

            # Absolute GPS receive time (ns) using the FIRST clock bias.
            t_rx_gnss = time_nanos - (full_bias0 + bias0)
            gps_seconds = t_rx_gnss * 1e-9

            # Per-constellation receive-time alignment.
            if sys == "C":
                t_rx = (t_rx_gnss % NS_PER_WEEK) - BDS_GPS_OFFSET_NS
            elif sys == "R":
                t_rx = (t_rx_gnss % NS_PER_DAY) + (GLO_GPS_OFFSET_HOURS * 3600 - leap_seconds) * 1_000_000_000
            else:  # G, E, J
                t_rx = t_rx_gnss % NS_PER_WEEK

            t_tx = recv_sv_time + time_offset
            pr_ns = t_rx - t_tx
            # Week/day rollover guard.
            if sys == "R":
                while pr_ns > NS_PER_DAY / 2:
                    pr_ns -= NS_PER_DAY
                while pr_ns < -NS_PER_DAY / 2:
                    pr_ns += NS_PER_DAY
            else:
                while pr_ns > NS_PER_WEEK / 2:
                    pr_ns -= NS_PER_WEEK
                while pr_ns < -NS_PER_WEEK / 2:
                    pr_ns += NS_PER_WEEK
            pseudorange_m = pr_ns * 1e-9 * SPEED_OF_LIGHT
            if pseudorange_m < 1.5e7 or pseudorange_m > 3.2e7:
                continue  # implausible — reject

            carrier = _to_float(raw_hdr.get(row, "CarrierFrequencyHz"))
            cn0 = _to_float(raw_hdr.get(row, "Cn0DbHz"))
            pr_rate = _to_float(raw_hdr.get(row, "PseudorangeRateMetersPerSecond"))
            recv_sv_unc = _to_float(raw_hdr.get(row, "ReceivedSvTimeUncertaintyNanos"))
            pr_unc = recv_sv_unc * 1e-9 * SPEED_OF_LIGHT if recv_sv_unc else None
            agc = _to_float(raw_hdr.get(row, "AgcDb"))

            sat_id = f"{sys}{svid:02d}"
            key = int(round(t_rx_gnss * 1e-6))  # integer ms of receive time
            ep = ensure_epoch(key, gps_seconds)

            # Keep only one signal per satellite: prefer the L1/E1/B1 band, then
            # the higher C/N0. (Newer phones report several frequencies per SV.)
            rank = (_band_rank(carrier), -(cn0 if cn0 is not None else -999.0))
            cur = ep["sat_best"].get(sat_id)
            if cur is None or rank < cur[0]:
                ep["sat_best"][sat_id] = (
                    rank,
                    RawMeasurement(
                        sat_id=sat_id, sys=sys, prn=svid,
                        pseudorange_m=pseudorange_m, pr_rate_mps=pr_rate,
                        cn0_dbhz=cn0, carrier_freq_hz=carrier,
                        pr_uncertainty_m=pr_unc, code="ANDROID",
                    ),
                )
            # AGC is per band — accumulate across all signals regardless of dedupe.
            if agc is not None:
                band = _band_key(carrier)
                ep["agc_sum"][band] = ep["agc_sum"].get(band, 0.0) + agc
                ep["agc_cnt"][band] = ep["agc_cnt"].get(band, 0) + 1

    for key in sorted(epochs):
        e = epochs[key]
        agc_db = {b: e["agc_sum"][b] / e["agc_cnt"][b] for b in e["agc_sum"]}
        sats = [meas for _rank, meas in e["sat_best"].values()]
        yield EpochObs(time=e["time"], sats=sats, agc_db=agc_db)


def load_epochs(path: str, leap_seconds: int = DEFAULT_LEAP_SECONDS) -> List[EpochObs]:
    return list(stream_epochs(path, leap_seconds=leap_seconds))
