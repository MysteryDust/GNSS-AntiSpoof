package com.gnssantispoof

import kotlin.math.sqrt

/** AGC/C-N₀ Receiver Power Monitoring — twin of `src/antispoof/physical.py`. */
enum class RfVerdict { NOMINAL, JAMMING, SPOOFING, UNKNOWN }

data class PhysicalReport(
    val verdict: RfVerdict,
    val agcDb: Double?, val agcBaseline: Double?, val agcDropDb: Double?,
    val agcAnomaly: Boolean, val cn0Median: Double?, val cn0Std: Double?,
    val cn0High: Boolean, val cn0Uniform: Boolean, val cn0Dropped: Boolean,
    val reasons: List<String>,
)

class PhysicalMonitor(
    private val baselineEpochs: Int = 8,
    private val agcKSigma: Double = 4.0,
    private val agcMinStepDb: Double = 3.0,
    private val cn0HighDbHz: Double = 50.0,
    private val cn0UniformStd: Double = 2.0,
    private val cn0DropDbHz: Double = 6.0,
) {
    private val agcHist = ArrayDeque<Double>()
    private val cn0BaseHist = ArrayList<Double>()
    private var agcBaseline: Double? = null
    private var agcBaselineStd = 0.5
    private var cn0BaselineMedian: Double? = null
    private var nSeen = 0
    // Sustained-anomaly debounce: an overpowered attack produces a *sustained* AGC
    // drop; live environments produce transient dips. Require persistence before
    // calling it spoofing/jamming to cut false alarms.
    private var anomStreak = 0
    private val sustainEpochs = 3

    private val baselineReady get() = agcBaseline != null || cn0BaselineMedian != null

    private fun median(v: List<Double>): Double? {
        if (v.isEmpty()) return null
        val s = v.sorted(); val n = s.size
        return if (n % 2 == 1) s[n / 2] else 0.5 * (s[n / 2 - 1] + s[n / 2])
    }

    private fun std(v: List<Double>): Double? {
        if (v.size < 2) return if (v.isEmpty()) null else 0.0
        val mean = v.average()
        return sqrt(v.sumOf { (it - mean) * (it - mean) } / (v.size - 1))
    }

    fun update(epoch: EpochObs): PhysicalReport {
        nSeen++
        val agc = epoch.meanAgc()
        val cn0Vals = epoch.sats.mapNotNull { it.cn0DbHz }
        val cn0Median = median(cn0Vals); val cn0Std = std(cn0Vals)
        val reasons = ArrayList<String>()

        if (!baselineReady || nSeen <= baselineEpochs) {
            if (agc != null) { agcHist.addLast(agc); if (agcHist.size > 30) agcHist.removeFirst() }
            if (cn0Median != null) cn0BaseHist.add(cn0Median)
            if (nSeen >= baselineEpochs) {
                if (agcHist.isNotEmpty()) {
                    agcBaseline = agcHist.average()
                    agcBaselineStd = maxOf(std(agcHist.toList()) ?: 0.5, 0.2)
                }
                if (cn0BaseHist.isNotEmpty()) cn0BaselineMedian = median(cn0BaseHist)
            }
            reasons.add("learning baseline")
            return PhysicalReport(RfVerdict.UNKNOWN, agc, agcBaseline, null,
                false, cn0Median, cn0Std, false, false, false, reasons)
        }

        var agcAnomaly = false; var agcDrop: Double? = null
        if (agc != null && agcBaseline != null) {
            agcDrop = agcBaseline!! - agc
            val step = maxOf(agcKSigma * agcBaselineStd, agcMinStepDb)
            if (agcDrop > step) { agcAnomaly = true; reasons.add("AGC dropped %.1f dB (added RF power)".format(agcDrop)) }
        }
        var cn0High = false; var cn0Dropped = false; var cn0Uniform = false
        if (cn0Median != null) {
            if (cn0Median > cn0HighDbHz) { cn0High = true; reasons.add("median C/N0 %.1f > %.0f".format(cn0Median, cn0HighDbHz)) }
            val base = cn0BaselineMedian
            if (base != null && cn0Median < base - cn0DropDbHz) { cn0Dropped = true; reasons.add("C/N0 dropped") }
        }
        if (cn0Std != null && cn0Vals.size >= 5 && cn0Std < cn0UniformStd) {
            cn0Uniform = true; reasons.add("C/N0 spread %.1f dB across %d SVs (single antenna)".format(cn0Std, cn0Vals.size))
        }

        anomStreak = if (agcAnomaly) anomStreak + 1 else 0
        val sustained = anomStreak >= sustainEpochs
        val verdict = when {
            agcAnomaly && sustained && cn0Dropped -> RfVerdict.JAMMING
            agcAnomaly && sustained -> RfVerdict.SPOOFING
            cn0High && cn0Uniform -> RfVerdict.SPOOFING
            cn0Uniform && (cn0Median ?: 0.0) > 45.0 -> RfVerdict.SPOOFING
            else -> RfVerdict.NOMINAL
        }
        return PhysicalReport(verdict, agc, agcBaseline, agcDrop, agcAnomaly,
            cn0Median, cn0Std, cn0High, cn0Uniform, cn0Dropped, reasons)
    }
}
