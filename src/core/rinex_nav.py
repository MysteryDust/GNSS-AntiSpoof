"""RINEX 3.04 mixed broadcast-navigation parser.

Supports GPS (G), Galileo (E) and BeiDou (C) Keplerian ephemeris (8 lines each)
and GLONASS (R) state-vector ephemeris (4 lines).

Returned structure:
    {
      'G': {prn: [Ephemeris, ...]},
      'E': {...},
      'C': {...},
      'R': {prn: [GlonassEph, ...]},
      'iono': {...optional...}
    }
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List


def _f(s: str):
    s = s.strip().replace("D", "E").replace("d", "E")
    if not s:
        return 0.0
    return float(s)


@dataclass
class KeplerEph:
    """Keplerian ephemeris record (GPS/Galileo/BeiDou). Field names follow standard."""
    sys: str
    prn: int
    toc: datetime          # time of clock (epoch from header line)
    af0: float
    af1: float
    af2: float
    # broadcast orbit 1
    iode: float
    crs: float
    delta_n: float
    m0: float
    # broadcast orbit 2
    cuc: float
    e: float
    cus: float
    sqrt_a: float
    # broadcast orbit 3
    toe: float             # time of ephemeris (seconds of GNSS week)
    cic: float
    omega0: float
    cis: float
    # broadcast orbit 4
    i0: float
    crc: float
    omega: float
    omega_dot: float
    # broadcast orbit 5
    idot: float
    codes_l2: float
    gnss_week: float
    l2p_flag: float
    # broadcast orbit 6
    sv_accuracy: float
    sv_health: float
    tgd: float
    iodc_or_bgd_e5a_e1: float
    # broadcast orbit 7
    transmission_time: float
    fit_interval: float
    spare1: float
    spare2: float


@dataclass
class GlonassEph:
    sys: str
    prn: int
    toc: datetime
    tau_n: float           # -clock bias
    gamma_n: float         # relative frequency bias
    tk: float              # message frame time
    x: float; vx: float; ax: float; health: float
    y: float; vy: float; ay: float; freq_num: float
    z: float; vz: float; az: float; age_op: float


def parse_rinex_nav(path):
    out: Dict[str, Dict[int, List]] = {"G": {}, "E": {}, "C": {}, "R": {}, "J": {}, "I": {}, "S": {}}
    iono = {}

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        # ---- header ----
        for line in fh:
            label = line[60:].rstrip()
            if label == "END OF HEADER":
                break
            if label == "IONOSPHERIC CORR":
                tag = line[:4].strip()
                vals = [_f(line[5 + 12 * i: 5 + 12 * (i + 1)]) for i in range(4)]
                iono.setdefault(tag, []).append(vals)
            elif label == "TIME SYSTEM CORR":
                pass  # ignored — we approximate constellation offsets via constants

        # ---- body ----
        while True:
            head = fh.readline()
            if not head:
                break
            if len(head) < 23:
                continue
            sys = head[0]
            try:
                prn = int(head[1:3])
            except ValueError:
                continue
            y = int(head[4:8]); m = int(head[9:11]); d = int(head[12:14])
            hh = int(head[15:17]); mm = int(head[18:20]); ss = int(head[21:23])
            toc = datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)
            af0 = _f(head[23:42]); af1 = _f(head[42:61]); af2 = _f(head[61:80])

            if sys == "R":
                lines = [fh.readline() for _ in range(3)]
                if any(not l for l in lines):
                    break
                eph = GlonassEph(
                    sys=sys, prn=prn, toc=toc,
                    tau_n=af0, gamma_n=af1, tk=af2,
                    x=_f(lines[0][4:23]), vx=_f(lines[0][23:42]),
                    ax=_f(lines[0][42:61]), health=_f(lines[0][61:80]),
                    y=_f(lines[1][4:23]), vy=_f(lines[1][23:42]),
                    ay=_f(lines[1][42:61]), freq_num=_f(lines[1][61:80]),
                    z=_f(lines[2][4:23]), vz=_f(lines[2][23:42]),
                    az=_f(lines[2][42:61]), age_op=_f(lines[2][61:80]),
                )
                out["R"].setdefault(prn, []).append(eph)
            elif sys in ("G", "E", "C", "J", "I"):
                lines = [fh.readline() for _ in range(7)]
                if any(not l for l in lines):
                    break
                def F(i, j):
                    return _f(lines[i][4 + 19 * j: 4 + 19 * (j + 1)])
                eph = KeplerEph(
                    sys=sys, prn=prn, toc=toc, af0=af0, af1=af1, af2=af2,
                    iode=F(0, 0), crs=F(0, 1), delta_n=F(0, 2), m0=F(0, 3),
                    cuc=F(1, 0), e=F(1, 1), cus=F(1, 2), sqrt_a=F(1, 3),
                    toe=F(2, 0), cic=F(2, 1), omega0=F(2, 2), cis=F(2, 3),
                    i0=F(3, 0), crc=F(3, 1), omega=F(3, 2), omega_dot=F(3, 3),
                    idot=F(4, 0), codes_l2=F(4, 1),
                    gnss_week=F(4, 2), l2p_flag=F(4, 3),
                    sv_accuracy=F(5, 0), sv_health=F(5, 1),
                    tgd=F(5, 2), iodc_or_bgd_e5a_e1=F(5, 3),
                    transmission_time=F(6, 0), fit_interval=F(6, 1),
                    spare1=F(6, 2), spare2=F(6, 3),
                )
                out.setdefault(sys, {}).setdefault(prn, []).append(eph)
            else:
                # unknown system — skip 7 lines defensively
                for _ in range(7):
                    if not fh.readline():
                        break

    return out, iono


def pick_ephemeris(eph_list, gps_seconds, max_dt=2 * 3600):
    """Return ephemeris record nearest to gps_seconds (absolute GPS seconds since epoch),
    or None if no record within max_dt."""
    if not eph_list:
        return None
    from .timeutils import datetime_to_gps_sow
    best = None
    best_dt = float("inf")
    for e in eph_list:
        toc_secs = datetime_to_gps_sow(e.toc, leap_seconds=0)
        dt = abs(gps_seconds - toc_secs)
        if dt < best_dt:
            best_dt = dt
            best = e
    if best_dt > max_dt:
        return None
    return best
