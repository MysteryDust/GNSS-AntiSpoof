"""Anti-spoofing engine behaviour tests (detection + mitigation)."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.antispoof.detector import SpoofStatus                                 # noqa: E402
from src.realtime.engine import RealtimeEngine                                 # noqa: E402
from src.simulate.scenario import (                                            # noqa: E402
    make_synthetic_constellation, make_trajectory, simulate_track,
)
from src.simulate.spoofer import SpoofConfig, apply_spoofing                   # noqa: E402

M_PER_DEG = 111_132.0
VISIBLE = ["G03", "G06", "G07", "G10", "G13", "G17", "G20", "E03", "E04", "E08", "E09", "E13", "E18"]


def _herr(o, t):
    dlat = (o.lat_deg - t.lat_deg) * M_PER_DEG
    dlon = (o.lon_deg - t.lon_deg) * M_PER_DEG * math.cos(math.radians(t.lat_deg))
    return math.sqrt(dlat * dlat + dlon * dlon)


def _setup(spoof, start=50, end=120, offset=(600, 200, 0), ramp=False, n=150):
    nav, ref = make_synthetic_constellation()
    traj = make_trajectory(n_epochs=n, ref_time=ref)
    isb = {"E": 12.0}
    clean = simulate_track(traj, nav, isb_true=isb, noise_m=0.6)
    cfg = SpoofConfig(target_prns=spoof, start_epoch=start, end_epoch=end,
                      false_offset_enu=offset, ramp=ramp)
    spoofed, meta = apply_spoofing(clean, traj, nav, cfg, isb_true=isb)
    engine = RealtimeEngine(nav, gate_m=120.0, physical_baseline_epochs=8)
    outputs = engine.run(spoofed)
    return outputs, traj, meta


def test_clean_run_has_no_false_alarms():
    nav, ref = make_synthetic_constellation()
    traj = make_trajectory(n_epochs=60, ref_time=ref)
    clean = simulate_track(traj, nav, isb_true={"E": 12.0}, noise_m=0.6)
    engine = RealtimeEngine(nav, physical_baseline_epochs=8)
    outputs = engine.run(clean)
    spoofed = [o for o in outputs if o.status == SpoofStatus.SPOOFED]
    assert not spoofed, f"{len(spoofed)} false spoof alarms on clean data"
    # And position is accurate.
    errs = [_herr(o, traj[i]) for i, o in enumerate(outputs) if o.lat_deg]
    assert max(errs) < 5.0


def test_minority_subset_detected_and_mitigated():
    outputs, traj, meta = _setup(["G06", "G10", "G17", "E04"], offset=(600, 200, 0))
    attack_idx = [i for i, a in enumerate(meta.active_by_epoch) if a]
    det = sum(1 for i in attack_idx if outputs[i].status == SpoofStatus.SPOOFED)
    assert det / len(attack_idx) > 0.95, "detection rate too low"
    # Mitigated error must be far below naive error during the attack.
    mitig = [_herr(outputs[i], traj[i]) for i in attack_idx if outputs[i].lat_deg]
    rms_m = math.sqrt(sum(e * e for e in mitig) / len(mitig))
    assert rms_m < 10.0, f"mitigated attack error {rms_m} m too high"
    # No false alarms on clean epochs.
    fa = sum(1 for i, a in enumerate(meta.active_by_epoch)
             if not a and outputs[i].status == SpoofStatus.SPOOFED)
    assert fa == 0, f"{fa} false alarms"


def test_identification_precision_no_authentic_blamed():
    outputs, traj, meta = _setup(["G06", "G10", "G17", "E04"], offset=(700, 250, 0))
    fp = 0
    for i, a in enumerate(meta.active_by_epoch):
        if not a:
            continue
        truth = set(meta.spoofed_prns_by_epoch[i])
        for prn in outputs[i].spoofed_prns:
            if prn not in truth:
                fp += 1
    assert fp == 0, f"{fp} authentic satellites wrongly named as spoofed"


def test_majority_spoof_triggers_fallback():
    outputs, traj, meta = _setup(
        ["G03", "G06", "G07", "G10", "G13", "E03", "E04", "E08"], offset=(500, 200, 0))
    attack_idx = [i for i, a in enumerate(meta.active_by_epoch) if a]
    det = sum(1 for i in attack_idx if outputs[i].status == SpoofStatus.SPOOFED)
    assert det / len(attack_idx) > 0.9
    # Even under majority spoof, position must stay protected (dead-reckoning).
    mitig = [_herr(outputs[i], traj[i]) for i in attack_idx if outputs[i].lat_deg]
    rms_m = math.sqrt(sum(e * e for e in mitig) / len(mitig))
    assert rms_m < 15.0, f"majority-spoof position not protected ({rms_m} m)"
    # The naive solution should be badly off (proving the attack was strong).
    naive = [math.hypot((outputs[i].naive_lat_deg - traj[i].lat_deg) * M_PER_DEG,
                        (outputs[i].naive_lon_deg - traj[i].lon_deg) * M_PER_DEG
                        * math.cos(math.radians(traj[i].lat_deg)))
             for i in attack_idx if outputs[i].naive_lat_deg]
    assert max(naive) > 100.0


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
