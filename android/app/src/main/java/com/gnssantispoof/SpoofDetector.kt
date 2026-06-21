package com.gnssantispoof

import kotlin.math.abs

enum class SpoofStatus { CLEAN, SUSPECT, SPOOFED }
enum class Confidence { LOW, MEDIUM, HIGH }

data class SpoofReport(
    val status: SpoofStatus, val confidence: Confidence,
    val spoofedPrns: List<String>, val excludedPrns: List<String>,
    val trustedFix: Fix?, val mitigated: Boolean, val naiveFix: Fix?,
    val reasons: List<String>, val nSats: Int,
)

/** Fuse RANSAC + RAIM + physical RPM — twin of `src/antispoof/detector.py`. */
class SpoofDetector(private val gateM: Double = 80.0) {
    private val bigResidualM = 30.0

    fun process(states: List<SatState>, physical: PhysicalReport?, positionPrior: DoubleArray?,
                gateOverrideM: Double? = null): SpoofReport {
        val n = states.size
        val prior = positionPrior ?: priorFrom(states)
        val naive = Solver.wlsPosition(states, prior)
        val gate = gateOverrideM ?: gateM
        val ransac = Ransac.run(states, prior, positionPrior, if (positionPrior != null) gate else null)

        if (!ransac.ok || ransac.fix == null) {
            val status = if (n >= 5) SpoofStatus.SUSPECT else SpoofStatus.CLEAN
            return SpoofReport(status, Confidence.LOW, emptyList(), emptyList(), null, true, naive,
                listOf("RANSAC failed: ${ransac.reason}"), n)
        }

        val excluded = ransac.excludedIds
        val inliers = states.filter { it.satId in ransac.inlierIds.toSet() }
        val inlierConsistent = Raim.consistency(inliers)
        val raimFull = Raim.fde(states)
        val maxExcl = excluded.maxOfOrNull { abs(ransac.residuals[it] ?: 0.0) } ?: 0.0
        val big = maxExcl > bigResidualM
        val confident = excluded.filter {
            abs(ransac.residuals[it] ?: 0.0) > maxOf(2.0 * (ransac.thresholds[it] ?: 20.0), bigResidualM)
        }
        val physSpoof = physical?.verdict == RfVerdict.SPOOFING
        val physJam = physical?.verdict == RfVerdict.JAMMING
        // Routine on real data (multipath) — not spoofing on its own.
        val geometricSubset = excluded.isNotEmpty() && inlierConsistent
        // Do the EXCLUDED satellites form their own consistent fix far away?
        // (single-antenna spoofed subset vs scattered multipath).
        val coherent = excludedFormCoherentFix(states, excluded, ransac.fix)
        val corroborated = physSpoof || coherent
        val reasons = ArrayList<String>()

        if (ransac.priorGateRejected) {
            reasons.add(ransac.reason)
            physical?.reasons?.let { reasons.addAll(it) }
            // Dead-reckon either way, but only *call it spoofing* when corroborated.
            return SpoofReport(
                if (corroborated) SpoofStatus.SPOOFED else SpoofStatus.SUSPECT,
                if (corroborated) Confidence.HIGH else Confidence.LOW,
                emptyList(), excluded, null, false, naive,
                reasons + "consensus drifted from prior; dead-reckoning", n)
        }
        if (geometricSubset && corroborated) {
            reasons.add("RANSAC isolated ${ransac.consensusSize} inliers; excluded $excluded")
            if (coherent) reasons.add("excluded satellites form a coherent fix elsewhere — spoofed subset")
            val conf = if (physSpoof) { physical?.reasons?.let { reasons.addAll(it) }; Confidence.HIGH } else Confidence.MEDIUM
            val named = if (confident.isNotEmpty()) confident else if (coherent) excluded else emptyList()
            return SpoofReport(SpoofStatus.SPOOFED, conf, named, excluded, ransac.fix, true, naive, reasons, n)
        }
        if (physSpoof || physJam) {
            physical?.reasons?.let { reasons.addAll(it) }
            reasons.add("RF anomaly without a coherent spoofed subset — flagging, trusting outlier-rejected fix")
            return SpoofReport(SpoofStatus.SUSPECT, if (physSpoof) Confidence.MEDIUM else Confidence.LOW,
                emptyList(), excluded, ransac.fix, true, naive, reasons, n)
        }
        if (geometricSubset && big) {
            reasons.add("excluded $excluded (residual up to %.1f m); likely multipath".format(maxExcl))
            return SpoofReport(SpoofStatus.SUSPECT, Confidence.LOW, emptyList(), excluded, ransac.fix, true, naive, reasons, n)
        }
        if (excluded.isNotEmpty() || raimFull.faultDetected) {
            reasons.add("routine outlier rejection; no spoofing indicators")
            return SpoofReport(SpoofStatus.CLEAN, Confidence.MEDIUM, emptyList(), excluded, ransac.fix, true, naive, reasons, n)
        }
        reasons.add("all consistency checks passed")
        return SpoofReport(SpoofStatus.CLEAN, Confidence.HIGH, emptyList(), excluded, ransac.fix, true, naive, reasons, n)
    }

    /** True if the excluded satellites form their own consistent fix far from the
     *  inlier fix — the signature of a single-antenna spoofed subset. */
    private fun excludedFormCoherentFix(states: List<SatState>, excluded: List<String>, inlierFix: Fix?): Boolean {
        if (inlierFix == null || excluded.size < 5) return false
        val es = states.filter { it.satId in excluded.toSet() }
        val fx = Solver.wlsPosition(es, inlierFix.ecef) ?: return false
        // Solver.Fix carries pdop but not rms; recompute a coarse residual check via residuals map.
        val rms = if (fx.residuals.isEmpty()) 1e9 else
            kotlin.math.sqrt(fx.residuals.values.sumOf { it * it } / fx.residuals.size)
        if (rms > 25.0) return false
        val a = fx.ecef; val b = inlierFix.ecef
        val sep = kotlin.math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]))
        return sep > 100.0
    }

    private fun priorFrom(states: List<SatState>): DoubleArray {
        if (states.isEmpty()) return doubleArrayOf(4438000.0, 3086000.0, 3375000.0)
        val s = states[0]
        val nrm = kotlin.math.sqrt(s.x * s.x + s.y * s.y + s.z * s.z)
        return doubleArrayOf(s.x / nrm * 6371000.0, s.y / nrm * 6371000.0, s.z / nrm * 6371000.0)
    }
}
