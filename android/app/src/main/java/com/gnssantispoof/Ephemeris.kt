package com.gnssantispoof

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Broadcast Keplerian ephemeris + WGS-84 coordinate maths — twin of
 * `src/core/ephemeris.py` and `src/core/coordinates.py`. (GLONASS RK4 omitted for brevity;
 * the engine defaults to GPS/Galileo/BeiDou, as the Python core does with SKIP_GLONASS.)
 */
data class KeplerEph(
    val sys: String, val prn: Int, val tocGpsSeconds: Double,
    val af0: Double, val af1: Double, val af2: Double,
    val crs: Double, val deltaN: Double, val m0: Double,
    val cuc: Double, val e: Double, val cus: Double, val sqrtA: Double,
    val toe: Double, val cic: Double, val omega0: Double, val cis: Double,
    val i0: Double, val crc: Double, val omega: Double, val omegaDot: Double,
    val idot: Double, val tgd: Double,
)

object Ephem {
    const val C = 2.99792458e8
    private const val MU = 3.986005e14
    private const val MU_BDS = 3.986004418e14
    private const val OMEGA_E = 7.2921151467e-5
    private const val OMEGA_E_BDS = 7.292115e-5
    private const val F_REL = -4.442807633e-10
    const val SECONDS_PER_WEEK = 604800.0

    private fun kepler(m: Double, e: Double): Double {
        var ea = m
        repeat(30) {
            val d = (ea - e * sin(ea) - m) / (1.0 - e * cos(ea)); ea -= d
            if (kotlin.math.abs(d) < 1e-12) return@repeat
        }
        return ea
    }

    fun svClockCorrection(eph: KeplerEph, tSow: Double): Double {
        var dt = tSow - eph.toe
        if (dt > 302400) dt -= 604800 else if (dt < -302400) dt += 604800
        return eph.af0 + eph.af1 * dt + eph.af2 * dt * dt
    }

    /** Returns [x,y,z, vx,vy,vz, relativisticCorrectionSeconds]. */
    fun computeSvEcef(eph: KeplerEph, tTransmitSow: Double): DoubleArray {
        val mu = if (eph.sys == Sys.BEIDOU) MU_BDS else MU
        val omegaE = if (eph.sys == Sys.BEIDOU) OMEGA_E_BDS else OMEGA_E
        val a = eph.sqrtA * eph.sqrtA
        val n0 = sqrt(mu / (a * a * a))
        var tk = tTransmitSow - eph.toe
        if (tk > 302400) tk -= 604800 else if (tk < -302400) tk += 604800
        val n = n0 + eph.deltaN
        val m = eph.m0 + n * tk
        val ea = kepler(m, eph.e)
        val sinE = sin(ea); val cosE = cos(ea)
        val sq = sqrt(1.0 - eph.e * eph.e)
        val nu = atan2(sq * sinE / (1 - eph.e * cosE), (cosE - eph.e) / (1 - eph.e * cosE))
        val phi = nu + eph.omega
        val s2 = sin(2 * phi); val c2 = cos(2 * phi)
        val u = phi + eph.cuc * c2 + eph.cus * s2
        val r = a * (1 - eph.e * cosE) + eph.crc * c2 + eph.crs * s2
        val inc = eph.i0 + eph.cic * c2 + eph.cis * s2 + eph.idot * tk
        val xOrb = r * cos(u); val yOrb = r * sin(u)
        val omegaK = eph.omega0 + (eph.omegaDot - omegaE) * tk - omegaE * eph.toe
        val so = sin(omegaK); val co = cos(omegaK); val si = sin(inc); val ci = cos(inc)
        val x = xOrb * co - yOrb * ci * so
        val y = xOrb * so + yOrb * ci * co
        val z = yOrb * si
        // Velocity
        val eDot = n / (1 - eph.e * cosE)
        val nuDot = sq * eDot / (1 - eph.e * cosE)
        val uDot = nuDot + 2 * (eph.cus * c2 - eph.cuc * s2) * nuDot
        val rDot = a * eph.e * sinE * eDot + 2 * (eph.crs * c2 - eph.crc * s2) * nuDot
        val iDot = eph.idot + 2 * (eph.cis * c2 - eph.cic * s2) * nuDot
        val xdo = rDot * cos(u) - r * sin(u) * uDot
        val ydo = rDot * sin(u) + r * cos(u) * uDot
        val odk = eph.omegaDot - omegaE
        val vx = xdo * co - ydo * ci * so + yOrb * si * so * iDot - (xOrb * so + yOrb * ci * co) * odk
        val vy = xdo * so + ydo * ci * co - yOrb * si * co * iDot + (xOrb * co - yOrb * ci * so) * odk
        val vz = ydo * si + yOrb * ci * iDot
        val rel = F_REL * eph.e * eph.sqrtA * sinE
        return doubleArrayOf(x, y, z, vx, vy, vz, rel)
    }

    fun applyEarthRotation(x: Double, y: Double, z: Double, transit: Double): DoubleArray {
        val th = OMEGA_E * transit; val c = cos(th); val s = sin(th)
        return doubleArrayOf(c * x + s * y, -s * x + c * y, z)
    }
}

object Wgs84 {
    private const val A = 6378137.0
    private const val F = 1.0 / 298.257223563
    private const val B = A * (1.0 - F)
    private const val E2 = 1.0 - (B * B) / (A * A)

    /** ECEF (m) -> [latRad, lonRad, altM]. */
    fun ecefToGeodetic(x: Double, y: Double, z: Double): DoubleArray {
        val ep2 = (A * A - B * B) / (B * B)
        val p = sqrt(x * x + y * y)
        if (p < 1e-9) return doubleArrayOf(if (z >= 0) Math.PI / 2 else -Math.PI / 2, 0.0, kotlin.math.abs(z) - B)
        val th = atan2(z * A, p * B)
        val lon = atan2(y, x)
        val lat = atan2(z + ep2 * B * sin(th) * sin(th) * sin(th),
            p - E2 * A * cos(th) * cos(th) * cos(th))
        val nn = A / sqrt(1 - E2 * sin(lat) * sin(lat))
        return doubleArrayOf(lat, lon, p / cos(lat) - nn)
    }

    fun elevationRad(rx: DoubleArray, sv: DoubleArray): Double {
        val g = ecefToGeodetic(rx[0], rx[1], rx[2]); val lat = g[0]; val lon = g[1]
        val dx = sv[0] - rx[0]; val dy = sv[1] - rx[1]; val dz = sv[2] - rx[2]
        val sl = sin(lat); val cl = cos(lat); val so = sin(lon); val co = cos(lon)
        val e = -so * dx + co * dy
        val nn = -sl * co * dx - sl * so * dy + cl * dz
        val up = cl * co * dx + cl * so * dy + sl * dz
        return atan2(up, sqrt(e * e + nn * nn))
    }
}
