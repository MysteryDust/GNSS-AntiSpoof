package com.gnssantispoof

import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/** Supplies broadcast ephemeris for a satellite at a GPS time. Provide by decoding
 *  GnssNavigationMessage on-device, or by parsing a downloaded BRDC file. */
interface EphemerisProvider {
    fun ephemerisFor(satId: String, gpsSeconds: Double): KeplerEph?
}

/** Per-satellite geometry+observable for one epoch — twin of solver.SatState. */
data class SatState(
    val satId: String, val sys: String, val prn: Int,
    val x: Double, val y: Double, val z: Double,
    val vx: Double, val vy: Double, val vz: Double,
    val svClockS: Double,
    val pseudorangeM: Double,
    val prRateMps: Double?,
    val cn0DbHz: Double?,
    var weight: Double = 1.0,
    val elevationRad: Double? = null,
    val prUncertaintyM: Double? = null,
)

data class Fix(
    val ecef: DoubleArray, val clockBiasM: Double,
    val isb: Map<String, Double>, val satIds: List<String>,
    val residuals: Map<String, Double>, val nSats: Int,
    val pdop: Double,
)

/** WLS PVT over satellite states — twin of `src/core/solver.py`. */
object Solver {
    const val C = 2.99792458e8
    private const val BDS_GPS_OFFSET = 14.0
    val SYS_ORDER = listOf("E", "C", "J")   // GPS reference; GLONASS omitted in this port

    fun computeSatStates(
        epoch: EpochObs, eph: EphemerisProvider,
        rxPrior: DoubleArray, elevationMaskRad: Double = Math.toRadians(10.0),
    ): List<SatState> {
        val week = (epoch.timeGpsSeconds / Ephem.SECONDS_PER_WEEK).toInt()
        val out = ArrayList<SatState>()
        val haveReal = rxPrior[0] * rxPrior[0] + rxPrior[1] * rxPrior[1] + rxPrior[2] * rxPrior[2] > 1e12
        for (m in epoch.sats) {
            if (m.sys == Sys.GLONASS) continue
            val e = eph.ephemerisFor(m.satId, epoch.timeGpsSeconds) ?: continue
            val tEmit = epoch.timeGpsSeconds - m.pseudorangeM / C
            val offset = if (m.sys == Sys.BEIDOU) -BDS_GPS_OFFSET else 0.0
            val tEmitSow = (tEmit + offset) - week * Ephem.SECONDS_PER_WEEK
            val clk0 = Ephem.svClockCorrection(e, tEmitSow)
            val sv = Ephem.computeSvEcef(e, tEmitSow - clk0)
            val svClock = clk0 + sv[6] - e.tgd
            val rot = Ephem.applyEarthRotation(sv[0], sv[1], sv[2], m.pseudorangeM / C)
            val dx = rot[0] - rxPrior[0]; val dy = rot[1] - rxPrior[1]; val dz = rot[2] - rxPrior[2]
            val rho = sqrt(dx * dx + dy * dy + dz * dz)
            if (rho < 1e3) continue
            var elev: Double? = null
            var w = 1.0
            if (haveReal) {
                val el = Wgs84.elevationRad(rxPrior, doubleArrayOf(rot[0], rot[1], rot[2]))
                if (el < elevationMaskRad) continue
                elev = el; w = max(sin(el), 0.05); w *= w
            }
            if (m.prUncertaintyM != null && m.prUncertaintyM > 0) {
                val sigma = maxOf(m.prUncertaintyM, 0.5)   // floor so one over-confident sat can't dominate
                w /= sigma * sigma
            }
            out.add(SatState(m.satId, m.sys, m.prn, rot[0], rot[1], rot[2],
                sv[3], sv[4], sv[5], svClock, m.pseudorangeM, m.prRateMps, m.cn0DbHz,
                w, elev, m.prUncertaintyM))
        }
        return out
    }

    fun activeSys(states: List<SatState>): List<String> {
        val seen = states.map { it.sys }.toSet()
        return SYS_ORDER.filter { it in seen }
    }

    fun wlsPosition(states: List<SatState>, prior: DoubleArray, maxIter: Int = 12, minSats: Int = 4): Fix? {
        if (states.size < minSats) return null
        val active = activeSys(states)
        val nUnk = 4 + active.size
        if (states.size < nUnk) return null
        val rx = prior.copyOf()
        var clk = 0.0
        val isb = HashMap<String, Double>().apply { active.forEach { put(it, 0.0) } }
        var converged = false
        repeat(maxIter) {
            val h = ArrayList<DoubleArray>(); val res = ArrayList<Double>(); val w = ArrayList<Double>()
            for (s in states) {
                val dx = s.x - rx[0]; val dy = s.y - rx[1]; val dz = s.z - rx[2]
                val rho = sqrt(dx * dx + dy * dy + dz * dz)
                val ux = dx / rho; val uy = dy / rho; val uz = dz / rho
                val extra = if (s.sys != "G") isb[s.sys] ?: 0.0 else 0.0
                val predicted = rho + clk + extra - C * s.svClockS
                val row = DoubleArray(nUnk)
                row[0] = -ux; row[1] = -uy; row[2] = -uz; row[3] = 1.0
                if (s.sys in active) row[4 + active.indexOf(s.sys)] = 1.0
                h.add(row); res.add(s.pseudorangeM - predicted); w.add(s.weight)
            }
            if (h.size < nUnk) return null
            val d = try {
                LinAlg.solveWls(h.toTypedArray(), res.toDoubleArray(), w.toDoubleArray())
            } catch (e: ArithmeticException) { return null }
            rx[0] += d[0]; rx[1] += d[1]; rx[2] += d[2]; clk += d[3]
            active.forEachIndexed { i, sy -> isb[sy] = (isb[sy] ?: 0.0) + d[4 + i] }
            if (kotlin.math.abs(d[0]) < 1e-3 && kotlin.math.abs(d[1]) < 1e-3 && kotlin.math.abs(d[2]) < 1e-3) {
                converged = true; return@repeat
            }
        }
        if (!converged) return null
        val residuals = HashMap<String, Double>()
        val hPos = ArrayList<DoubleArray>()
        for (s in states) {
            val dx = s.x - rx[0]; val dy = s.y - rx[1]; val dz = s.z - rx[2]
            val rho = sqrt(dx * dx + dy * dy + dz * dz)
            val extra = if (s.sys != "G") isb[s.sys] ?: 0.0 else 0.0
            residuals[s.satId] = s.pseudorangeM - (rho + clk + extra - C * s.svClockS)
            hPos.add(doubleArrayOf(-dx / rho, -dy / rho, -dz / rho, 1.0))
        }
        val pdop = try { computePdop(hPos) } catch (e: Exception) { 99.0 }
        return Fix(rx, clk, isb, states.map { it.satId }, residuals, states.size, pdop)
    }

    private fun computePdop(hPos: List<DoubleArray>): Double {
        val hth = Array(4) { DoubleArray(4) }
        for (row in hPos) for (i in 0..3) for (j in 0..3) hth[i][j] += row[i] * row[j]
        val inv = LinAlg.invert(hth)
        return sqrt(max(inv[0][0] + inv[1][1] + inv[2][2], 0.0))
    }

    fun wlsVelocity(states: List<SatState>, rx: DoubleArray): DoubleArray? {
        val rows = ArrayList<DoubleArray>(); val obs = ArrayList<Double>()
        for (s in states) {
            val pr = s.prRateMps ?: continue
            val dx = s.x - rx[0]; val dy = s.y - rx[1]; val dz = s.z - rx[2]
            val rho = sqrt(dx * dx + dy * dy + dz * dz)
            val ux = dx / rho; val uy = dy / rho; val uz = dz / rho
            val svProj = s.vx * ux + s.vy * uy + s.vz * uz
            rows.add(doubleArrayOf(-ux, -uy, -uz, 1.0)); obs.add(pr - svProj)
        }
        if (rows.size < 4) return null
        return try {
            LinAlg.solveWls(rows.toTypedArray(), obs.toDoubleArray(), DoubleArray(rows.size) { 1.0 })
        } catch (e: ArithmeticException) { null }
    }
}
