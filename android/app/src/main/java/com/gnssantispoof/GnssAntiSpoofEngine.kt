package com.gnssantispoof

import android.location.GnssMeasurementsEvent
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import kotlin.math.sqrt

/** One epoch of engine output, exposed to the UI. */
data class EngineOutput(
    val gpsSeconds: Double,
    val status: SpoofStatus, val confidence: Confidence, val nSats: Int,
    val spoofedPrns: List<String>,
    val latDeg: Double?, val lonDeg: Double?, val altM: Double?,
    val speedMps: Double?, val pdop: Double?,
    val naiveLatDeg: Double?, val naiveLonDeg: Double?,
    val rfVerdict: RfVerdict?, val deadReckoned: Boolean,
    val reasons: List<String>,
)

/**
 * Real-time standalone-positioning + anti-spoofing engine — twin of
 * `src/realtime/engine.py`. Register [measurementsCallback] with
 * `LocationManager.registerGnssMeasurementsCallback` and observe [output].
 */
class GnssAntiSpoofEngine(
    private val ephemeris: EphemerisProvider,
    private val leapSeconds: Int = 18,
    private val elevationMaskDeg: Double = 10.0,
    gateM: Double = 120.0,
    baselineEpochs: Int = 8,
) {
    private val reconstructor = PseudorangeReconstructor()
    private val physicalMonitor = PhysicalMonitor(baselineEpochs)
    private val detector = SpoofDetector(gateM)

    private var priorPos: DoubleArray? = null
    private var priorVel: DoubleArray? = null
    private var priorTime: Double? = null
    private var lastTrustedTime: Double? = null
    private var haveTrusted = false
    private val seed = doubleArrayOf(4438000.0, 3086000.0, 3375000.0)

    private val gateM = gateM
    private val gateGrowthMps = 25.0
    private val gateMaxM = 2000.0
    private val reacquireAfterS = 30.0

    private val _output = MutableLiveData<EngineOutput>()
    val output: LiveData<EngineOutput> = _output

    val measurementsCallback = object : GnssMeasurementsEvent.Callback() {
        override fun onGnssMeasurementsReceived(event: GnssMeasurementsEvent) {
            val epoch = reconstructor.process(event, leapSeconds) ?: return
            _output.postValue(process(epoch))
        }
    }

    private fun predictPrior(time: Double): Pair<DoubleArray?, Double?> {
        val p = priorPos ?: return null to null
        val pt = priorTime; val v = priorVel
        if (pt == null || v == null) return p to null
        val dt = time - pt
        if (dt <= 0 || dt >= 30) return p to dt
        return doubleArrayOf(p[0] + v[0] * dt, p[1] + v[1] * dt, p[2] + v[2] * dt) to dt
    }

    fun process(epoch: EpochObs): EngineOutput {
        val phys = physicalMonitor.update(epoch)
        val (predicted, _) = predictPrior(epoch.timeGpsSeconds)
        val rxPrior = predicted ?: priorPos ?: seed
        val states = Solver.computeSatStates(epoch, ephemeris, rxPrior, Math.toRadians(elevationMaskDeg))

        if (states.size < 5) {
            return deadReckon(epoch, predicted, phys, SpoofStatus.CLEAN, emptyList(),
                listOf("only ${states.size} usable satellites"))
        }

        // Time-aware prior gate: widen with dead-reckoning age while the RF
        // environment is clean and re-acquire after a sustained benign outage, but
        // keep it tight and engaged while an attack is actively flagged.
        var gatePrior: DoubleArray? = null
        var gate = gateM
        val attackOngoing = phys.verdict == RfVerdict.SPOOFING || phys.verdict == RfVerdict.JAMMING
        val ltt = lastTrustedTime
        if (haveTrusted && ltt != null) {
            val drAge = epoch.timeGpsSeconds - ltt
            when {
                attackOngoing -> { gatePrior = predicted; gate = gateM }
                drAge <= reacquireAfterS -> {
                    gatePrior = predicted
                    gate = minOf(gateM + gateGrowthMps * maxOf(drAge, 0.0), gateMaxM)
                }
                else -> priorVel = null   // benign sustained outage: re-acquire
            }
        }
        val report = detector.process(states, phys, gatePrior, gate)

        val naive = report.naiveFix?.let { Wgs84.ecefToGeodetic(it.ecef[0], it.ecef[1], it.ecef[2]) }
        val trusted = report.trustedFix
        if (trusted == null || !report.mitigated) {
            return deadReckon(epoch, predicted, phys, report.status, report.spoofedPrns, report.reasons,
                naive?.let { Math.toDegrees(it[0]) }, naive?.let { Math.toDegrees(it[1]) })
        }

        val g = Wgs84.ecefToGeodetic(trusted.ecef[0], trusted.ecef[1], trusted.ecef[2])
        val inliers = states.filter { it.satId in trusted.satIds.toSet() }
        val vel = Solver.wlsVelocity(inliers, trusted.ecef)
        val speed = vel?.let { sqrt(it[0] * it[0] + it[1] * it[1] + it[2] * it[2]) }

        priorPos = trusted.ecef
        priorVel = if (vel != null) doubleArrayOf(vel[0], vel[1], vel[2]) else null
        priorTime = epoch.timeGpsSeconds
        lastTrustedTime = epoch.timeGpsSeconds
        haveTrusted = true

        return EngineOutput(
            epoch.timeGpsSeconds, report.status, report.confidence, states.size, report.spoofedPrns,
            Math.toDegrees(g[0]), Math.toDegrees(g[1]), g[2], speed, trusted.pdop,
            naive?.let { Math.toDegrees(it[0]) }, naive?.let { Math.toDegrees(it[1]) },
            phys.verdict, false, report.reasons,
        )
    }

    private fun deadReckon(
        epoch: EpochObs, predicted: DoubleArray?, phys: PhysicalReport,
        status: SpoofStatus, spoofed: List<String>, reasons: List<String>,
        naiveLat: Double? = null, naiveLon: Double? = null,
    ): EngineOutput {
        val ecef = predicted
        var lat: Double? = null; var lon: Double? = null; var alt: Double? = null
        if (ecef != null) {
            val g = Wgs84.ecefToGeodetic(ecef[0], ecef[1], ecef[2])
            lat = Math.toDegrees(g[0]); lon = Math.toDegrees(g[1]); alt = g[2]
            priorPos = ecef; priorTime = epoch.timeGpsSeconds
        }
        return EngineOutput(epoch.timeGpsSeconds, status, Confidence.MEDIUM, epoch.sats.size,
            spoofed, lat, lon, alt, null, null, naiveLat, naiveLon, phys.verdict, ecef != null,
            reasons + if (ecef != null) listOf("dead-reckoned from last trusted fix") else emptyList())
    }
}
