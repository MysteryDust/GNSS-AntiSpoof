"""RINEX 3/4 observation-file parser.

Returns a list of Epoch objects. Each Epoch holds a GPS-time timestamp and a
dict {sat_id -> {obs_type -> value}} where sat_id is the 3-char system+PRN
(e.g. 'G08', 'E11', 'C20', 'R03').
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List


@dataclass
class Epoch:
    time: datetime  # GPS time, timezone-aware UTC equivalent
    flag: int
    obs: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _safe_float(s: str):
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_rinex_obs(path):
    """Parse a RINEX 3.x or 4.x OBS file. Returns (header_dict, [Epoch, ...])."""
    header = {"obs_types": {}}  # system letter -> list of obs codes
    epochs: List[Epoch] = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        # ---- header ----
        current_sys = None
        pending_types: list = []
        pending_count = 0
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading header")
            label = line[60:].rstrip()
            if label == "END OF HEADER":
                break
            if label == "RINEX VERSION / TYPE":
                header["version"] = float(line[:9].strip())
                header["type"] = line[20]
                header["sys"] = line[40]
            elif label == "SYS / # / OBS TYPES":
                # Continuation lines have a blank leading character.
                if line[0] != " ":
                    current_sys = line[0]
                    pending_count = int(line[3:6])
                    pending_types = line[7:60].split()
                else:
                    pending_types += line[7:60].split()
                if len(pending_types) >= pending_count:
                    header["obs_types"][current_sys] = pending_types[:pending_count]
                    current_sys = None
                    pending_types = []
            elif label == "TIME OF FIRST OBS":
                y = int(line[0:6]); m = int(line[6:12]); d = int(line[12:18])
                hh = int(line[18:24]); mm = int(line[24:30]); ss = float(line[30:43])
                header["time_first_obs"] = datetime(
                    y, m, d, hh, mm, int(ss),
                    int(round((ss - int(ss)) * 1e6)),
                    tzinfo=timezone.utc,
                )
                header["time_system"] = line[48:51].strip()
            elif label == "INTERVAL":
                header["interval"] = float(line[:10].strip() or 0.0)

        # ---- body ----
        for raw in fh:
            if not raw or raw[0] != ">":
                continue
            y = int(raw[2:6]); m = int(raw[7:9]); d = int(raw[10:12])
            hh = int(raw[13:15]); mm = int(raw[16:18])
            ss = float(raw[19:29])
            flag = int(raw[31:32])
            n_sat = int(raw[32:35])
            micro = int(round((ss - int(ss)) * 1e6)) % 1_000_000
            t = datetime(y, m, d, hh, mm, int(ss), micro, tzinfo=timezone.utc)
            epoch = Epoch(time=t, flag=flag)
            for _ in range(n_sat):
                sat_line = fh.readline()
                if not sat_line:
                    break
                sat_id = sat_line[:3]
                sys = sat_id[0]
                types = header["obs_types"].get(sys, [])
                values: Dict[str, float] = {}
                for i, tcode in enumerate(types):
                    start = 3 + i * 16
                    chunk = sat_line[start:start + 14]
                    v = _safe_float(chunk)
                    if v is not None:
                        values[tcode] = v
                if values:
                    epoch.obs[sat_id] = values
            epochs.append(epoch)

    return header, epochs
