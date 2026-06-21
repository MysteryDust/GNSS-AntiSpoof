"""Core solver, statistics and time-handling tests."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.antispoof.stats import chi2_isf                                       # noqa: E402
from src.core.solver import compute_sat_states, wls_position, wls_velocity     # noqa: E402
from src.simulate.scenario import (                                            # noqa: E402
    make_synthetic_constellation, make_trajectory, simulate_track, velocity_enu_to_ecef,
)


def test_chi2_isf_matches_table():
    # Standard chi-square inverse-survival values.
    cases = [(1e-5, 4, 28.473), (1e-3, 3, 16.266), (1e-2, 1, 6.635),
             (1e-5, 5, 30.856), (1e-6, 8, 42.701)]
    for pfa, dof, expect in cases:
        assert abs(chi2_isf(pfa, dof) - expect) < 0.01, (pfa, dof)


def test_solver_recovers_truth_noise_free():
    nav, ref = make_synthetic_constellation()
    traj = make_trajectory(n_epochs=3, ref_time=ref)
    epochs = simulate_track(traj, nav, isb_true={"E": 12.0}, noise_m=0.0)
    truth = traj[0]
    states = compute_sat_states(epochs[0], nav, rx_prior=truth.ecef)
    fix = wls_position(states, prior=truth.ecef)
    assert fix is not None
    err = math.sqrt(sum((fix["ecef"][i] - truth.ecef[i]) ** 2 for i in range(3)))
    assert err < 0.01, f"noise-free position error {err} m too large"
    assert abs(fix["isb"].get("E", 0.0) - 12.0) < 0.05
    assert abs(fix["clock_bias_m"] - truth.clock_bias_m) < 0.05


def test_velocity_recovered():
    nav, ref = make_synthetic_constellation()
    traj = make_trajectory(n_epochs=3, ref_time=ref, speed_mps=8.0, heading_deg=90.0)
    epochs = simulate_track(traj, nav, isb_true={"E": 12.0}, noise_m=0.0,
                            speed_mps=8.0, heading_deg=90.0)
    truth = traj[0]
    states = compute_sat_states(epochs[0], nav, rx_prior=truth.ecef)
    fix = wls_position(states, prior=truth.ecef)
    vel = wls_velocity([s for s in states if s.sat_id in fix["satellites"]], fix["ecef"])
    assert vel is not None
    truth_v = velocity_enu_to_ecef(8.0, 90.0, truth.lat_deg, truth.lon_deg)
    verr = math.sqrt(sum((vel[k] - truth_v[i]) ** 2 for i, k in enumerate(("vx", "vy", "vz"))))
    # The simulator injects ~0.03 m/s Doppler noise per satellite, so a few
    # cm/s residual is expected; assert it is well within a decimetre/s.
    assert verr < 0.1, f"velocity error {verr} m/s too large"


def test_beidou_isb_recovered():
    # Exercise the BeiDou (-14 s offset) path and a second inter-system bias.
    nav, ref = make_synthetic_constellation(n_bds=12)
    traj = make_trajectory(n_epochs=2, ref_time=ref)
    isb_true = {"E": 12.0, "C": -7.0}
    epochs = simulate_track(traj, nav, isb_true=isb_true, noise_m=0.0)
    truth = traj[0]
    ep = epochs[0]
    assert any(m.sys == "C" for m in ep.sats), "no BeiDou satellites visible"
    states = compute_sat_states(ep, nav, rx_prior=truth.ecef)
    fix = wls_position(states, prior=truth.ecef)
    assert fix is not None
    err = math.sqrt(sum((fix["ecef"][i] - truth.ecef[i]) ** 2 for i in range(3)))
    assert err < 0.02, f"position error with BeiDou {err} m"
    assert abs(fix["isb"].get("E", 0.0) - 12.0) < 0.1
    assert abs(fix["isb"].get("C", 0.0) - (-7.0)) < 0.1


def test_good_geometry():
    nav, ref = make_synthetic_constellation()
    traj = make_trajectory(n_epochs=1, ref_time=ref)
    epochs = simulate_track(traj, nav, noise_m=0.0)
    assert epochs[0].n_sats >= 10
    states = compute_sat_states(epochs[0], nav, rx_prior=traj[0].ecef)
    fix = wls_position(states, prior=traj[0].ecef)
    assert fix["dop"]["pdop"] < 3.0


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
