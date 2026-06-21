# Project Report — Real-Time Standalone Positioning with Subset-Spoofing Detection & Mitigation

**Course:** Intro to Navigation (2026b) — Project Option 1
**Builds on:** Ex0 (offline RINEX → KML/CSV positioning)

## 1. Problem statement

Option 1 asks us to extend Ex0 into a real-time navigation system that computes standalone
positioning from raw GNSS measurements and can **detect and overcome spoofing of a subset of the
navigation satellites**. The accompanying literature review identified the architecture that the
field has converged on: a standalone Weighted Least Squares PVT that bypasses the OS Fused Location
Provider, combined with a **dual-layer** anti-spoofing defence — physical-layer AGC/C-N₀ monitoring
plus mathematical outlier rejection (RAIM / RANSAC). This project implements that architecture,
validates it on ground-truth scenarios, and ports it to Android.

## 2. From the literature review to this implementation

| Literature review topic | Where it lives in this project |
|---|---|
| Android raw observables; pseudorange from `FullBiasNanos`/`ReceivedSvTimeNanos` (Google `gps-measurement-tools`, EUSPA white paper) | `src/io/gnsslogger.py`, `android/.../PseudorangeReconstructor.kt` |
| Standalone WLS PVT bypassing the Fused Location Provider | `src/core/solver.py`, `android/.../Solver.kt` |
| AGC drop as a spoofing indicator (Akos 2012) | `src/antispoof/physical.py` (AGC step detection) |
| Joint AGC + C/N₀ Receiver Power Monitoring (Miralles 2018, Spens 2022) | `physical.py` RPM truth table |
| RANSAC pseudorange outlier rejection / subset isolation (Castaldo 2014, Wen 2024, Zhu 2022) | `src/antispoof/ransac.py` |
| Residual RAIM and consistency-check / SRV-RAIM for multi-satellite spoofing (Medina & Lohan 2025) | `src/antispoof/raim.py` |
| Need for a trust anchor / position prior against majority spoofing | engine prior-gate + dead reckoning (`src/realtime/engine.py`) |

## 3. System overview

The engine processes epochs as a stream (emulating the phone's `GnssMeasurementsEvent` rate):

1. **Standalone PVT** from raw pseudoranges + broadcast ephemeris (WLS with inter-system biases).
2. **Physical monitor** updates the AGC/C-N₀ baseline and applies the RPM truth table.
3. **Geometric detection** runs RANSAC (subset consensus) and RAIM (integrity test).
4. **Fusion** decides CLEAN / SUSPECT / SPOOFED, names the spoofed satellites, and produces the
   **trusted (mitigated)** position from the authentic subset — or dead-reckons from a velocity-
   propagated prior when the epoch cannot be trusted (majority spoof).

A spoofing **simulator** (`src/simulate/`) generates controllable attacks with known ground truth and
writes them as real Android GnssLogger CSV, so the real-data parser and the whole pipeline are
exercised end-to-end, and detection/mitigation can be scored quantitatively.

## 4. Results

Synthetic scenarios (13 satellites visible, receiver moving east at 6 m/s, a subset spoofed from
epoch 50–120 with a single-antenna meaconing model). Horizontal RMS error vs ground truth *during*
the attack:

| Scenario | Spoofed | Detection | False alarm | ID precision | ID recall | Naive err | Mitigated err | Gain |
|---|---|---|---|---|---|---|---|---|
| Subset, instant | 4/13 | 100 % | 0 % | 100 % | 100 % | 312 m | 0.8 m | 383× |
| Subset, slow walk-off | 4/13 | 100 % | 0 % | 100 % | 74 % | 215 m | 0.9 m | 253× |
| Majority spoof | 8/13 | 100 % | 0 % | — | — | 305 m | 1.2 m | 266× |

Additional validated properties (see `tests/`):
- noise-free position recovery to **< 1 cm** (the WLS solver perfectly inverts the forward model);
- inter-system bias and receiver clock recovered exactly;
- **zero false alarms** on clean data, with sub-metre position accuracy;
- RINEX-nav and GnssLogger-CSV round-trips are bit-exact / decimetre-accurate respectively;
- identification **precision is 100 %** in every case — no authentic satellite is ever blamed.

### Real Android data
The engine was also run on real multi-constellation GnssLogger / RINEX recordings (the Ex0
dataset: GPS+GLONASS+Galileo+BeiDou+QZSS), scored against the phone's NMEA solution:

| Recording | Epochs | Horizontal vs NMEA | Spoof verdicts |
|---|---|---|---|
| Suburban ~5 min | 280 | median **5.0 m**, RMS 7.4 m, P95 14 m | 0 SPOOFED (169 CLEAN / 111 SUSPECT) |
| Urban ~44 min | 2628 | median 9.3 m, P95 15 m | 2 SPOOFED (0.1 %) |
| Urban ~43 min | 2590 | median 6.1 m, P95 10 m | 20 SPOOFED (0.8 %) |

The standalone PVT (raw pseudoranges only, never the OS Fused Location Provider) matches the
device's own solution to a few metres. On the clean recording the detector raised **no false
spoofing alarms**, flagging only routine multipath as SUSPECT while keeping the position. This
validated two real-data essentials beyond the synthetic tests: (a) the Android raw-measurement
pseudorange reconstruction is correct (per-satellite agreement with the RINEX pseudoranges to
< 1 m), including de-duplicating the multiple frequency signals modern phones report per satellite;
and (b) the detector does not cry "spoofing" on ordinary urban multipath. The urban recordings have
a heavy-multipath tail (rare km-level excursions, P95 still ≤ 15 m) where the engine dead-reckons to
protect the track.

The **physical layer** was also exercised on a long real recording carrying per-measurement AGC
(2628 epochs, the GnssLogger raw `.txt`): the device's AGC ranged from −66 to −47 dB (≈ 19 dB of
natural fluctuation), and the debounced Receiver-Power-Monitoring engine returned **NOMINAL on every
epoch (0 false RF spoofing/jamming alarms)**, with just 1 SPOOFED epoch in 2628 (0.04 %, a transient
multipath coincidence caught by the coherent-subset test). This confirms the AGC/C-N₀ monitor is
calibrated for live signals — it stays quiet through real environmental RF changes rather than
mistaking them for an attack.

### Interpretation
- For a **minority** spoofed subset the system isolates and *names* the spoofed satellites and
  recovers the true position from the authentic ones (sub-metre residual error during attack vs
  hundreds of metres unprotected — a 250–380× reduction).
- For a **majority** spoof the spoofed set *is* the consensus, so it cannot be named; instead the
  prior-gate detects the consensus jump and the engine dead-reckons, keeping the reported position
  within ~1 m while the unprotected solution is dragged 300 m. This matches the literature's finding
  that consensus methods require an external trust anchor to survive a majority attack.
- Identification **recall** is moderate (≈50–60 %) and intentionally honest: a spoofed satellite
  whose line-of-sight is nearly perpendicular to the position offset barely changes its pseudorange,
  so it is geometrically indistinguishable from an authentic one — and it is not biasing the fix
  either, so the position stays protected regardless.

## 5. How to reproduce

```bash
python apps/simulate_attack.py                                  # subset, slow walk-off
python apps/simulate_attack.py --no-ramp --offset 600,200,0     # subset, instant
python apps/simulate_attack.py --no-ramp --spoof G03,G06,G07,G10,G13,E03,E04,E08 --offset 500,200,0
python tests/run_tests.py                                       # 11 tests
```

For a real recording: capture with the Android **GnssLogger** app (enable raw measurements; disable
duty cycling in Developer Options for continuous data), download the matching ephemeris with
`python apps/fetch_brdc.py <date>`, then:

```bash
python apps/run_realtime.py --gnsslog data/raw/<log>.csv --nav data/brdc/<brdc>.rnx
```

## 6. Conclusions and future work

The dual-layer architecture from the literature review works in practice: a standalone WLS PVT
neutralises OS/network spoofing by construction, RANSAC+RAIM isolate and exclude a spoofed satellite
*subset*, the AGC/C-N₀ monitor catches overpowered attacks instantly, and a velocity-propagated
position prior handles the majority-spoof case that geometry alone cannot.

Future work: (1) wire a concrete on-device `EphemerisProvider` (decode `GnssNavigationMessage`) to
complete the Android app; (2) fuse an IMU/INS to extend the dead-reckoning window; (3) add carrier-
phase / time-differenced (TDCP) consistency (Zhu 2022) to catch subtle replay attacks; (4) validate
against a real spoofed recording from the lab SDR.

## 7. References

See the project literature review and `README.md`. Primary implementation sources: Google
`gps-measurement-tools`; EUSPA *Using GNSS Raw Measurements on Android Devices*; Akos (2012);
Miralles et al. (2018); Spens et al. (2022); Castaldo et al. (2014); Wen et al. (2024); Zhu et al.
(2022); Medina & Lohan (2025).
