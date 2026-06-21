# GNSS-AntiSpoof — Android module

A Kotlin port of the verified Python engine, wired to Android's **raw GNSS measurement** API.
It computes a **standalone** PVT from `GnssMeasurementsEvent` (never the Fused Location Provider),
detects spoofing of a satellite subset with the same RANSAC + RAIM + AGC/C-N₀ logic, and exposes a
**trusted** location alongside the OS location for comparison.

> Status: this is a faithful, documented port intended to be opened and finished in Android Studio.
> It is **not compiled in this repo** (the Python core is the verified reference implementation).
> Every algorithm file states the Python module it mirrors so the two stay in lock-step.

## Why raw measurements (not the Fused Location Provider)

The OS `FusedLocationProvider` blends GNSS with Wi-Fi/cell/Bluetooth and accepts mock locations, so
it is vulnerable to OS-level and network spoofing. Reading `GnssMeasurementsEvent` and solving PVT
ourselves bypasses all of that — only the physical RF observables are trusted, and physical-layer
spoofing is then caught by the AGC/C-N₀ + RANSAC/RAIM layers.

## Mapping to the Python reference

| Kotlin file | Mirrors (Python) | Role |
|---|---|---|
| `PseudorangeReconstructor.kt` | `src/io/gnsslogger.py` | `GnssMeasurement` → pseudorange (State filter, per-constellation time base) |
| `Ephemeris.kt` | `src/core/ephemeris.py` | Keplerian SV position/velocity/clock |
| `Solver.kt` + `LinearAlgebra.kt` | `src/core/solver.py` | WLS PVT + inter-system bias, DOP, velocity |
| `Ransac.kt` | `src/antispoof/ransac.py` | consensus subset isolation |
| `Raim.kt` + `Chi2.kt` | `src/antispoof/raim.py` + `stats.py` | residual integrity FD/FDE |
| `PhysicalMonitor.kt` | `src/antispoof/physical.py` | AGC/C-N₀ Receiver Power Monitoring |
| `SpoofDetector.kt` | `src/antispoof/detector.py` | fusion → status + named PRNs |
| `GnssAntiSpoofEngine.kt` | `src/realtime/engine.py` | streaming engine + Android callbacks |

## Key Android APIs used

- `LocationManager.registerGnssMeasurementsCallback()` → `GnssMeasurementsEvent`
- `GnssMeasurement`: `state`, `svid`, `constellationType`, `receivedSvTimeNanos`,
  `timeOffsetNanos`, `cn0DbHz`, `pseudorangeRateMetersPerSecond`, and
  `getAutomaticGainControlLevelDb()` (API 26+) for AGC.
- `GnssClock`: `timeNanos`, `fullBiasNanos`, `biasNanos`, `leapSecond`.
- For comparison only: `FusedLocationProviderClient` / `LocationManager.GPS_PROVIDER`.

Manifest needs `ACCESS_FINE_LOCATION` (and the runtime permission). To get continuous carrier phase
and avoid duty-cycling gaps, disable *"Force full GNSS measurements"* duty cycling in Developer
Options on supported devices.

## Broadcast ephemeris

SV positions need broadcast ephemeris. Provide it through `EphemerisProvider` — either by decoding
the navigation messages from `GnssNavigationMessage` callbacks on-device, or by downloading the
day's mixed BRDC file (see `apps/fetch_brdc.py`) and parsing it. A stub provider interface is
included; wiring a concrete provider is the one piece left for on-device deployment.

## Usage sketch

```kotlin
val engine = GnssAntiSpoofEngine(ephemerisProvider)
val lm = getSystemService(LOCATION_SERVICE) as LocationManager
lm.registerGnssMeasurementsCallback(engine.measurementsCallback, Handler(Looper.getMainLooper()))

engine.output.observe(this) { o ->
    statusView.text = "${o.status} (${o.confidence})  spoofed: ${o.spoofedPrns}"
    if (o.latDeg != null) map.show(o.latDeg, o.lonDeg)   // trusted, mitigated position
}
```
