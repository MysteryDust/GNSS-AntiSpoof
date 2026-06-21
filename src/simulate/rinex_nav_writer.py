"""Write a synthetic nav_data dict as a RINEX 3.04 navigation file.

Used by the demo/tests so the real-data CLI (``apps/run_realtime.py``) can be
exercised end-to-end on synthetic data: the simulator's broadcast ephemeris is
written to a RINEX nav file and parsed back by the same parser that reads a real
BRDC file. With a real recording you would instead download the matching BRDC
ephemeris (see ``apps/fetch_brdc.py``).
"""

from __future__ import annotations

from typing import Dict, List

from ..core.rinex_nav import KeplerEph


def _fmt(x: float) -> str:
    """19-char RINEX D-notation float field."""
    s = f"{x: .12E}".replace("E", "D")
    return s


def _epoch_line(eph: KeplerEph) -> str:
    t = eph.toc
    prefix = (f"{eph.sys}{eph.prn:02d} {t.year:04d} {t.month:02d} {t.day:02d} "
              f"{t.hour:02d} {t.minute:02d} {t.second:02d}")
    return prefix + _fmt(eph.af0) + _fmt(eph.af1) + _fmt(eph.af2)


def _orbit_line(*vals: float) -> str:
    return "    " + "".join(_fmt(v) for v in vals)


def write_rinex_nav(nav_data: Dict, path: str) -> None:
    lines: List[str] = []
    lines.append(f"{'3.04':>9}           N: GNSS NAV DATA    M: MIXED"
                 f"            RINEX VERSION / TYPE")
    lines.append(f"{'GNSS-AntiSpoof simulator':<20}{'':<20}{'':<20}COMMENT")
    lines.append(f"{'':<60}END OF HEADER")

    for sys in ("G", "E", "C", "J", "R"):
        for prn in sorted(nav_data.get(sys, {})):
            for eph in nav_data[sys][prn]:
                if sys == "R":
                    continue  # GLONASS state-vector records not emitted by the simulator
                lines.append(_epoch_line(eph))
                lines.append(_orbit_line(eph.iode, eph.crs, eph.delta_n, eph.m0))
                lines.append(_orbit_line(eph.cuc, eph.e, eph.cus, eph.sqrt_a))
                lines.append(_orbit_line(eph.toe, eph.cic, eph.omega0, eph.cis))
                lines.append(_orbit_line(eph.i0, eph.crc, eph.omega, eph.omega_dot))
                lines.append(_orbit_line(eph.idot, eph.codes_l2, eph.gnss_week, eph.l2p_flag))
                lines.append(_orbit_line(eph.sv_accuracy, eph.sv_health, eph.tgd,
                                         eph.iodc_or_bgd_e5a_e1))
                lines.append(_orbit_line(eph.transmission_time, eph.fit_interval,
                                         eph.spare1, eph.spare2))

    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")
