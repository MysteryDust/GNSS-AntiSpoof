package com.gnssantispoof

import android.location.GnssClock
import android.location.GnssMeasurement
import android.location.GnssMeasurementsEvent
import android.os.Build

/**
 * Reconstruct pseudoranges from Android raw measurements — the Kotlin twin of
 * `src/io/gnsslogger.py`. Same maths as Google's gps-measurement-tools / the EUSPA
 * white paper:
 *
 *   tRxGnss = TimeNanos - (FullBiasNanos0 + BiasNanos0)     // absolute GPS ns
 *   tRx     = tRxGnss mod week  (BeiDou -14 s, GLONASS time-of-day)
 *   tTx     = ReceivedSvTimeNanos + TimeOffsetNanos
 *   pr_m    = (tRx - tTx) * 1e-9 * c
 *
 * A measurement is used only when its State bitmask marks the pseudorange unambiguous.
 */
class PseudorangeReconstructor {

    companion object {
        const val C = 2.99792458e8
        const val NS_PER_WEEK = 604800L * 1_000_000_000L
        const val NS_PER_DAY = 86400L * 1_000_000_000L
        const val BDS_GPS_OFFSET_NS = 14L * 1_000_000_000L
        const val GLO_OFFSET_HOURS = 3

        // GnssMeasurement.State bits.
        const val STATE_CODE_LOCK = 0x1
        const val STATE_TOW_DECODED = 0x8
        const val STATE_GLO_TOD_DECODED = 0x80
        const val STATE_GAL_E1C_2ND_CODE_LOCK = 0x800
    }

    // The first clock bias seen in the session — used for all epochs (Google convention).
    private var fullBias0: Long? = null
    private var bias0: Double = 0.0

    private fun stateValid(state: Int, sys: String): Boolean = when (sys) {
        Sys.GLONASS -> (state and STATE_CODE_LOCK) != 0 && (state and STATE_GLO_TOD_DECODED) != 0
        // Require TOW decoded for GPS/Galileo/BeiDou/QZSS. The Galileo E1C 2nd-code
        // bit alone does not yield the full week-referenced pseudorange this parser
        // computes, so it is not accepted in isolation (100 ms ambiguity risk).
        else -> (state and STATE_CODE_LOCK) != 0 && (state and STATE_TOW_DECODED) != 0
    }

    private fun bandKey(carrierHz: Double?): String {
        val mhz = (carrierHz ?: 1_575_420_000.0) / 1e6
        return when {
            mhz in 1565.0..1612.0 -> "L1"
            mhz in 1160.0..1300.0 -> "L5"
            else -> "${Math.round(mhz)}MHz"
        }
    }

    /** Convert one GnssMeasurementsEvent into an EpochObs, or null if nothing usable. */
    fun process(event: GnssMeasurementsEvent, leapSeconds: Int = 18): EpochObs? {
        val clock: GnssClock = event.clock
        if (!clock.hasFullBiasNanos()) return null
        if (fullBias0 == null) {
            fullBias0 = clock.fullBiasNanos
            bias0 = if (clock.hasBiasNanos()) clock.biasNanos else 0.0
        }
        val fb0 = fullBias0!!
        val timeNanos = clock.timeNanos
        val tRxGnss = timeNanos - (fb0 + bias0.toLong())
        val gpsSeconds = tRxGnss * 1e-9

        val agcSum = HashMap<String, Double>()
        val agcCnt = HashMap<String, Int>()
        val sats = ArrayList<RawMeasurement>()

        for (m: GnssMeasurement in event.measurements) {
            val sys = Sys.fromConstellationType(m.constellationType) ?: continue
            if (!stateValid(m.state, sys)) continue

            val tRx: Long = when (sys) {
                Sys.BEIDOU -> (tRxGnss % NS_PER_WEEK) - BDS_GPS_OFFSET_NS
                Sys.GLONASS -> (tRxGnss % NS_PER_DAY) +
                    (GLO_OFFSET_HOURS * 3600L - leapSeconds) * 1_000_000_000L
                else -> tRxGnss % NS_PER_WEEK
            }
            val tTx = m.receivedSvTimeNanos + m.timeOffsetNanos.toLong()
            var prNs = tRx - tTx
            val rollover = if (sys == Sys.GLONASS) NS_PER_DAY else NS_PER_WEEK
            while (prNs > rollover / 2) prNs -= rollover
            while (prNs < -rollover / 2) prNs += rollover
            val prM = prNs * 1e-9 * C
            if (prM < 1.5e7 || prM > 3.2e7) continue

            val carrier = if (m.hasCarrierFrequencyHz()) m.carrierFrequencyHz.toDouble() else null
            val prUnc = m.receivedSvTimeUncertaintyNanos * 1e-9 * C
            val agc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
                m.hasAutomaticGainControlLevelDb()
            ) m.automaticGainControlLevelDb else null

            sats.add(
                RawMeasurement(
                    satId = "%s%02d".format(sys, m.svid),
                    sys = sys, prn = m.svid,
                    pseudorangeM = prM,
                    prRateMps = m.pseudorangeRateMetersPerSecond,
                    cn0DbHz = m.cn0DbHz,
                    carrierFreqHz = carrier,
                    prUncertaintyM = prUnc,
                )
            )
            if (agc != null) {
                val b = bandKey(carrier)
                agcSum[b] = (agcSum[b] ?: 0.0) + agc
                agcCnt[b] = (agcCnt[b] ?: 0) + 1
            }
        }
        if (sats.isEmpty()) return null
        val agcDb = agcSum.mapValues { (k, v) -> v / agcCnt[k]!! }
        return EpochObs(gpsSeconds, sats, agcDb)
    }
}
