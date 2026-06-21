"""Round-trip tests: simulator -> GnssLogger CSV / RINEX nav -> parsers -> solver."""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.solver import compute_sat_states, wls_position                  # noqa: E402
from src.io import gnsslogger, rinex_stream                                   # noqa: E402
from src.simulate.gnsslogger_writer import write_gnsslogger_csv               # noqa: E402
from src.simulate.rinex_nav_writer import write_rinex_nav                     # noqa: E402
from src.simulate.scenario import (                                           # noqa: E402
    make_synthetic_constellation, make_trajectory, simulate_track,
)


def test_gnsslogger_roundtrip_recovers_position():
    nav, ref = make_synthetic_constellation()
    traj = make_trajectory(n_epochs=5, ref_time=ref)
    clean = simulate_track(traj, nav, isb_true={"E": 12.0}, noise_m=0.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.csv")
        write_gnsslogger_csv(clean, path)
        parsed = gnsslogger.load_epochs(path)
    assert len(parsed) == len(clean)
    # Integer-nanosecond quantisation in the CSV limits accuracy to ~decimetre.
    truth = traj[0]
    states = compute_sat_states(parsed[0], nav, rx_prior=truth.ecef)
    fix = wls_position(states, prior=truth.ecef)
    err = math.sqrt(sum((fix["ecef"][i] - truth.ecef[i]) ** 2 for i in range(3)))
    assert err < 0.5, f"round-trip position error {err} m"


def test_rinex_nav_roundtrip_is_exact():
    nav, ref = make_synthetic_constellation()
    traj = make_trajectory(n_epochs=3, ref_time=ref)
    epochs = simulate_track(traj, nav, noise_m=0.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nav.rnx")
        write_rinex_nav(nav, path)
        nav2 = rinex_stream.load_nav(path)
    # SV positions from the original and round-tripped ephemerides must match.
    s1 = {s.sat_id: s for s in compute_sat_states(epochs[0], nav, rx_prior=traj[0].ecef)}
    s2 = {s.sat_id: s for s in compute_sat_states(epochs[0], nav2, rx_prior=traj[0].ecef)}
    assert set(s1) == set(s2) and len(s1) >= 10
    for sid in s1:
        d = math.sqrt(sum((getattr(s1[sid], a) - getattr(s2[sid], a)) ** 2 for a in ("x", "y", "z")))
        assert d < 1e-3, f"{sid} SV position differs by {d} m after RINEX round-trip"


def test_state_filter_rejects_invalid():
    # A measurement whose State lacks TOW_DECODED must be dropped.
    from src.io.gnsslogger import _state_valid
    assert _state_valid(0x1 | 0x8, "G")
    assert not _state_valid(0x1, "G")            # code lock only, no TOW
    assert _state_valid(0x1 | 0x80, "R")         # GLONASS TOD decoded
    assert not _state_valid(0x8, "R")            # GLONASS needs TOD bit, not TOW


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
