package com.gnssantispoof

/**
 * Unified GNSS measurement model — the Kotlin twin of `src/core/measurements.py`.
 * Both the live Android path and any file replay normalise into these types.
 */

/** RINEX-style constellation letters. */
object Sys {
    const val GPS = "G"
    const val GALILEO = "E"
    const val BEIDOU = "C"
    const val GLONASS = "R"
    const val QZSS = "J"
    const val SBAS = "S"
    const val IRNSS = "I"

    /** android.location.GnssStatus constellation type -> RINEX letter. */
    fun fromConstellationType(t: Int): String? = when (t) {
        1 -> GPS; 2 -> SBAS; 3 -> GLONASS; 4 -> QZSS; 5 -> BEIDOU; 6 -> GALILEO; 7 -> IRNSS
        else -> null
    }
}

/** One satellite's observation at one epoch. */
data class RawMeasurement(
    val satId: String,
    val sys: String,
    val prn: Int,
    val pseudorangeM: Double,
    val prRateMps: Double? = null,        // range rate (m/s), +ve = receding
    val cn0DbHz: Double? = null,
    val carrierFreqHz: Double? = null,
    val prUncertaintyM: Double? = null,
    val code: String = "ANDROID",
)

/** All measurements at one epoch plus receiver-level RF metrics. */
data class EpochObs(
    val timeGpsSeconds: Double,             // absolute GPS seconds since 1980-01-06
    val sats: List<RawMeasurement>,
    val agcDb: Map<String, Double> = emptyMap(),   // band -> AGC level (dB)
) {
    fun meanAgc(): Double? = agcDb.values.takeIf { it.isNotEmpty() }?.average()
    fun meanCn0(): Double? = sats.mapNotNull { it.cn0DbHz }.takeIf { it.isNotEmpty() }?.average()
}
