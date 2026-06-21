"""Adapt the Ex0 RINEX observation parser into the unified measurement model.

``parse_rinex_obs`` (ported from Ex0) yields ``Epoch`` objects keyed by RINEX
observation codes (C1C, C2I, D1C, ...). This module turns those into
:class:`~src.core.measurements.EpochObs` streams of :class:`RawMeasurement`,
selecting one pseudorange + one Doppler per satellite and converting the
Doppler (Hz) into a metric range rate so the unified solver can use it.

RINEX carries no AGC, so ``EpochObs.agc_db`` is left empty for this source —
the physical-layer spoofing monitor simply has nothing to act on and the
engine falls back to the geometric (RANSAC/RAIM) detectors.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

from ..core.ephemeris import C
from ..core.measurements import EpochObs, RawMeasurement
from ..core.rinex_nav import parse_rinex_nav
from ..core.rinex_obs import parse_rinex_obs
from ..core.solver import carrier_freq_for, _glonass_freq


def _pick_pseudorange(sys: str, obs: Dict[str, float]) -> Tuple[Optional[float], Optional[str]]:
    if sys == "C":
        order = ("C2I", "C1I", "C5Q", "C7I", "C1C")
    else:
        order = ("C1C", "C1X", "C5Q", "C5X")
    for c in order:
        if c in obs:
            return obs[c], c
    return None, None


def _pick_doppler(obs: Dict[str, float], code_used: Optional[str]) -> Optional[float]:
    """Pick a Doppler observable on the SAME band as the chosen pseudorange.

    Mixing a Doppler from a different frequency band would pair it with the wrong
    carrier wavelength in the range-rate conversion, so we only fall back to other
    Doppler codes that share the pseudorange's band digit (the char after 'C').
    """
    if code_used and code_used.startswith("C"):
        cand = "D" + code_used[1:]
        if cand in obs:
            return obs[cand]
        band = code_used[1]  # frequency-band digit, e.g. '1', '2', '5'
        for c in obs:
            if c.startswith("D" + band):
                return obs[c]
        return None
    # No pseudorange band known — accept any single-frequency Doppler.
    for c in ("D1C", "D1X", "D1I"):
        if c in obs:
            return obs[c]
    return None


def read_glonass_slots(obs_path: str) -> Dict[int, int]:
    """Parse the GLONASS SLOT / FRQ # header records (slot -> frequency channel k)."""
    slot_k: Dict[int, int] = {}
    with open(obs_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            label = line[60:].rstrip()
            if label == "END OF HEADER":
                break
            if label == "GLONASS SLOT / FRQ #":
                tokens = line[4:60].split()
                for i in range(0, len(tokens) - 1, 2):
                    tag = tokens[i]
                    if tag.startswith("R") and tag[1:].isdigit():
                        try:
                            slot_k[int(tag[1:])] = int(tokens[i + 1])
                        except ValueError:
                            pass
    return slot_k


def load_nav(nav_path: str):
    """Parse a RINEX navigation file into the nav_data structure the solver expects."""
    nav_data, _iono = parse_rinex_nav(nav_path)
    return nav_data


def stream_epochs(obs_path: str) -> Iterator[EpochObs]:
    """Yield :class:`EpochObs` for each epoch in a RINEX observation file."""
    header, epochs = parse_rinex_obs(obs_path)
    slot_k = read_glonass_slots(obs_path)
    for ep in epochs:
        sats: List[RawMeasurement] = []
        for sat_id, vals in ep.obs.items():
            sys = sat_id[0]
            try:
                prn = int(sat_id[1:])
            except ValueError:
                continue
            pr, code = _pick_pseudorange(sys, vals)
            if pr is None or pr <= 1e6 or pr > 5e7:
                continue
            slot = slot_k.get(prn) if sys == "R" else None
            cf = carrier_freq_for(sat_id, _glonass_freq(slot) if slot is not None else None)
            doppler = _pick_doppler(vals, code)
            # Doppler (Hz, +ve = approaching) -> range rate (m/s, +ve = receding).
            pr_rate = (-doppler * C / cf) if (doppler is not None and cf) else None
            sats.append(
                RawMeasurement(
                    sat_id=sat_id, sys=sys, prn=prn,
                    pseudorange_m=pr,
                    pr_rate_mps=pr_rate,
                    cn0_dbhz=None,
                    carrier_freq_hz=cf,
                    code=code,
                )
            )
        yield EpochObs(time=ep.time, sats=sats, agc_db={})


def load_epochs(obs_path: str) -> List[EpochObs]:
    return list(stream_epochs(obs_path))
