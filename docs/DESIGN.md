# Design & Algorithms

This document explains the algorithms behind GNSS-AntiSpoof and the design decisions that make
them work. It assumes the background in the project literature review.

## 1. Standalone PVT from raw measurements

### 1.1 Reconstructing pseudoranges (Android)
Android exposes the receiver's raw hardware clock rather than finished pseudoranges. We follow
Google's `gps-measurement-tools` / the EUSPA white paper. Using the first `FullBiasNanos` in the
session as the GPS-week reference (so the time line is continuous):

```
tRxGnss  = TimeNanos − (FullBiasNanos₀ + BiasNanos₀)          # absolute GPS nanoseconds
tRx      = tRxGnss mod WEEK            (GPS/Galileo/QZSS)
         = (tRxGnss mod WEEK) − 14 s   (BeiDou: BDT = GPST−14 s)
         = (tRxGnss mod DAY) + (3 h − leap)   (GLONASS: time-of-day → GPST)
tTx      = ReceivedSvTimeNanos + TimeOffsetNanos              # apply offset exactly once
pr [m]   = (tRx − tTx) · 10⁻⁹ · c
```

A measurement is used only when its `State` bitmask proves the pseudorange is unambiguous:
`STATE_CODE_LOCK (0x1)` **and** `STATE_TOW_DECODED (0x8)` for GPS/Galileo/BeiDou/QZSS, or
`STATE_GLO_TOD_DECODED (0x80)` for GLONASS. `C/N0` and `AutomaticGainControlLevelDb` are carried
through for the physical-layer monitor. (RINEX OBS files give pseudoranges directly; both sources
normalise into the same `EpochObs` model.)

### 1.2 Satellite positions and the WLS solve
Broadcast ephemeris gives each satellite's ECEF position/velocity and clock (Keplerian propagation
for GPS/Galileo/BeiDou, RK4 for GLONASS), with relativistic, group-delay (TGD) and Earth-rotation
(Sagnac) corrections. The predicted pseudorange for receiver state (**x**, clock `cδt`, inter-system
bias `b_sys`) is

```
ρ̂ᵢ = ‖SVᵢ − x‖ + cδt + b_sysᵢ − c·(SV clockᵢ)
```

and we solve the over-determined system by **weighted least squares**, iterating

```
Δ = (HᵀWH)⁻¹ HᵀW (ρ − ρ̂),   wᵢ = sin²(elᵢ)/σ²
```

The unknowns are receiver ECEF (x, y, z), the GPS clock, and **one inter-system bias per non-GPS
constellation** present (Galileo, BeiDou, QZSS) — these absorb the constellation time-scale and
receiver hardware offsets, which is essential when mixing constellations. Velocity and clock drift
come from a second linear solve on the Doppler/pseudorange-rate observables.

Because the position is derived *only* from RF observables and broadcast ephemeris, the Android OS
Fused Location Provider is never consulted — mock-location injection and Wi-Fi/network spoofing have
no effect on the computed position.

## 2. Detecting spoofing of a subset

### 2.1 RANSAC consensus (the subset isolator)
A single-antenna spoofer that fakes a *subset* of satellites makes that subset mutually consistent
with a false position, while the authentic satellites remain consistent with the true position.
RANSAC exploits exactly this:

1. **Whiten to one clock.** Estimate the inter-system biases once from a robust full solve, subtract
   them from every pseudorange, and force a single clock unknown. This keeps the **minimal subset at
   4** across all constellations — without it, each extra constellation adds an ISB unknown, the
   minimal subset grows, and the tolerable spoof fraction collapses.
2. **Hypothesise.** For each minimal 4-satellite subset (enumerated when `C(m,4) ≤ 1500`, otherwise
   randomly sampled `N = ⌈log(1−p)/log(1−wˢ)⌉` times with online shrinking of `N`), solve PVT.
3. **Score consensus.** Count satellites whose residual against that candidate is within an
   **elevation-weighted gate** `k·σ/sin(el)` (k=4). Keep the largest, lowest-cost consensus set.
4. **Refit + report.** Recompute the trusted fix by full WLS over the inliers (re-estimating the
   ISBs); the excluded set is the spoofed subset.

### 2.2 RAIM (the integrity test)
Classical residual RAIM provides the formal alarm and a self-consistency check:

```
SSE = wᵀ W w   (postfit weighted residual norm) ~ χ²(n − m)   under H₀
fault if  SSE > χ²₍₁₋Pfa₎(n − m),   Pfa = 10⁻⁵ per epoch
```

`χ²` thresholds are computed at runtime from a dependency-free regularised-incomplete-gamma
implementation (validated against the standard table, e.g. χ²₍₁₋10⁻⁵₎(4) = 28.473). Fault
Detection-and-Exclusion iteratively removes the satellite with the largest *normalised* residual
`|wᵢ|·√Wᵢ /√Sᵢᵢ` (S = I − hat matrix) and re-tests while redundancy remains. The engine uses RAIM
two ways: a fault flag on the full set, and a consistency check confirming the RANSAC inlier set is
internally clean.

### 2.3 Physical-layer Receiver Power Monitoring
Independent of geometry, an overpowered attack leaves an RF signature. We learn an open-sky AGC and
C/N0 baseline from the first clean epochs, then apply the RPM truth table:

| AGC vs baseline | C/N0 | verdict |
|---|---|---|
| unchanged | normal | NOMINAL |
| drop (≥ max(4σ, 3 dB)) | drops | JAMMING |
| drop | stable / high | **SPOOFING** |
| unchanged | abnormally high (>50 dB-Hz) **and** uniform (σ<2 dB-Hz across SVs) | **SPOOFING** (single antenna) |

AGC polarity/level is device-specific, so we detect **step changes** from baseline, not absolute
levels. These C/N0 numeric thresholds are practical heuristics (tunable per device), not hard
constants from any one paper.

## 3. Overcoming it — fusion and the trust anchor

`SpoofDetector` combines the three layers:

- **Geometric isolation** (RANSAC excluded a subset, inliers RAIM-consistent, corroborated by a RAIM
  fault or a large excluded residual) ⇒ **SPOOFED**, spoofed PRNs named, trusted PVT = inlier fix.
  Physical corroboration raises the confidence to HIGH.
- **Physical-only** (RPM says SPOOFING but the offset is too small to separate a subset) ⇒ **SPOOFED**
  but `mitigated = false`: the engine does **not** trust the possibly-biased all-satellite fix and
  dead-reckons instead.
- **Prior-gate rejection** (the consensus position itself jumps beyond the gate from the propagated
  prior) ⇒ **majority spoof**: no fix is trusted; dead-reckon and raise a hard alarm.

The **engine** (`realtime/engine.py`) provides the trust anchor RANSAC needs: it keeps the last
trusted position **propagated by velocity** every epoch (so the prior never goes stale even through a
long attack) and gates each new consensus against it. Only a geometrically-certified clean-subset fix
(or a clean epoch) advances the trusted prior; physical-only and majority-spoof epochs dead-reckon.

## 4. Known limitations (stated honestly)

- **Majority spoof cannot be *identified*.** When the spoofed satellites are the majority, they form
  the largest consensus, so RANSAC/RAIM cannot name them. The prior-gate + dead-reckoning keeps the
  position protected, but per-satellite identification recall is 0 in that regime — this is inherent
  to consensus methods, not a bug.
- **Perpendicular spoofing is invisible — and harmless.** A spoofed satellite whose line-of-sight is
  nearly perpendicular to the position offset barely changes its pseudorange, so it cannot be
  distinguished from an authentic one. It also barely biases the fix, so leaving it in is acceptable.
- **Physical detection needs the AGC field.** Some devices/Android versions report AGC per band in
  separate records or not at all; without it the engine relies on the geometric layer (and the C/N0
  shape cues) alone.
- **Dead reckoning drifts.** The fallback assumes roughly constant velocity; a long majority-spoof on
  a manoeuvring receiver will accumulate error. The prior gate widens with dead-reckoning age and, once
  the RF environment looks clean again, disengages after ~30 s so the engine re-acquires instead of
  locking onto a stale prior forever — but it deliberately stays tight and engaged while the physical
  layer is still flagging an attack (re-acquiring mid-spoof would walk onto the spoof). A sustained
  majority spoof that leaves *no* physical signature (e.g. RINEX input with no AGC) is the residual
  blind spot; fusing an IMU/INS would extend the safe window.
- **Slow walk-off has a recall tail.** Early in a gradual lift-off the offset is below the geometric
  gate, so the spoofed PRNs are not yet named (the physical layer still flags the attack and the
  position stays protected).
