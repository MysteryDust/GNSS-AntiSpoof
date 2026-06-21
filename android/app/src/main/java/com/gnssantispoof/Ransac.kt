package com.gnssantispoof

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * RANSAC pseudorange consensus — twin of `src/antispoof/ransac.py`. Estimates the
 * inter-system biases once, whitens to a single-clock frame (minimal subset = 4),
 * then keeps the largest self-consistent satellite set; the rest are the spoofed subset.
 */
data class RansacResult(
    val ok: Boolean, val fix: Fix?,
    val inlierIds: List<String> = emptyList(),
    val excludedIds: List<String> = emptyList(),
    val residuals: Map<String, Double> = emptyMap(),
    val thresholds: Map<String, Double> = emptyMap(),
    val consensusSize: Int = 0,
    val priorGateRejected: Boolean = false,
    val reason: String = "",
)

object Ransac {
    private const val P = 0.99
    private const val W0 = 0.6
    private const val K_SIGMA = 4.0
    private const val SIGMA_RHO = 5.0
    private const val MIN_SUBSET = 4
    private const val ENUMERATE_CAP = 1500
    private const val COARSE_GATE = 120.0

    private fun iterations(p: Double, w: Double, s: Int): Int {
        val ww = min(max(w, 1e-3), 0.999)
        val denom = ln(1.0 - Math.pow(ww, s.toDouble()))
        return if (denom == 0.0) 1 else max(1, ceil(ln(1.0 - p) / denom).toInt())
    }

    private fun whiten(states: List<SatState>, isb: Map<String, Double>): List<SatState> =
        states.map { s ->
            s.copy(sys = "G", pseudorangeM = s.pseudorangeM - (isb[s.sys] ?: 0.0))
        }

    private fun predictResidual(s: SatState, rx: DoubleArray, clk: Double): Double {
        val dx = s.x - rx[0]; val dy = s.y - rx[1]; val dz = s.z - rx[2]
        val rho = sqrt(dx * dx + dy * dy + dz * dz)
        return s.pseudorangeM - (rho + clk - Solver.C * s.svClockS)
    }

    private fun gate(s: SatState): Double {
        val sinEl = if (s.elevationRad != null) max(sin(s.elevationRad), 0.1) else 1.0
        return K_SIGMA * SIGMA_RHO / sinEl
    }

    private fun comb(n: Int, k: Int): Long {
        if (k > n) return 0
        var r = 1L; for (i in 0 until k) r = r * (n - i) / (i + 1); return r
    }

    fun run(
        states: List<SatState>, prior: DoubleArray,
        positionPrior: DoubleArray? = null, gateM: Double? = null, seed: Long = 12345L,
    ): RansacResult {
        val m = states.size
        if (m < MIN_SUBSET + 1) return RansacResult(false, null, reason = "too few satellites")

        // Coarse robust ISB estimate.
        var coarse = Solver.wlsPosition(states, prior) ?: return RansacResult(false, null, reason = "coarse solve failed")
        val kept = states.filter { abs(coarse.residuals[it.satId] ?: 0.0) <= COARSE_GATE }
        if (kept.size in MIN_SUBSET until states.size) {
            Solver.wlsPosition(kept, coarse.ecef)?.let { coarse = it }
        }
        val isb = HashMap<String, Double>().apply {
            put("G", 0.0); put("E", 0.0); put("C", 0.0); put("J", 0.0); putAll(coarse.isb)
        }
        val w = whiten(states, isb)
        val gates = w.map { gate(it) }

        val rng = java.util.Random(seed)
        val total = comb(m, MIN_SUBSET)
        val exhaustive = total <= ENUMERATE_CAP
        var maxIter = if (exhaustive) total.toInt() else iterations(P, W0, MIN_SUBSET)
        val subsets = if (exhaustive) allCombinations(m, MIN_SUBSET) else null

        var best = IntArray(0); var bestCost = Double.MAX_VALUE; var iter = 0
        while (iter < maxIter) {
            val idx = if (exhaustive) subsets!![iter] else randomSubset(rng, m, MIN_SUBSET)
            iter++
            val sub = idx.map { w[it] }
            val fix = Solver.wlsPosition(sub, prior, minSats = MIN_SUBSET) ?: continue
            val inliers = ArrayList<Int>(); var cost = 0.0
            for (j in w.indices) {
                val r = predictResidual(w[j], fix.ecef, fix.clockBiasM)
                if (abs(r) < gates[j]) { inliers.add(j); cost += r * r }
            }
            if (inliers.size < MIN_SUBSET + 1) continue
            if (inliers.size > best.size || (inliers.size == best.size && cost < bestCost)) {
                best = inliers.toIntArray(); bestCost = cost
                if (!exhaustive) maxIter = min(maxIter, iterations(P, inliers.size.toDouble() / m, MIN_SUBSET))
            }
        }
        if (best.isEmpty()) return RansacResult(false, null, consensusSize = 0, reason = "no consensus")

        val inlierIds = best.map { w[it].satId }.toSet()
        val finalStates = states.filter { it.satId in inlierIds }
        val finalFix = Solver.wlsPosition(finalStates, prior)
            ?: return RansacResult(false, null, reason = "final WLS failed")

        val residuals = HashMap<String, Double>(); val thresholds = HashMap<String, Double>()
        val refined = HashMap<String, Double>().apply {
            put("G", 0.0); put("E", 0.0); put("C", 0.0); put("J", 0.0); putAll(finalFix.isb)
        }
        for (j in states.indices) {
            val s = states[j]
            val ws = s.copy(sys = "G", pseudorangeM = s.pseudorangeM - (refined[s.sys] ?: 0.0))
            residuals[s.satId] = predictResidual(ws, finalFix.ecef, finalFix.clockBiasM)
            thresholds[s.satId] = gates[j]
        }
        val excluded = states.map { it.satId }.filter { it !in inlierIds }
        var priorReject = false; var reason = "ok"
        if (positionPrior != null && gateM != null) {
            val dx = finalFix.ecef[0] - positionPrior[0]
            val dy = finalFix.ecef[1] - positionPrior[1]
            val dz = finalFix.ecef[2] - positionPrior[2]
            val jump = sqrt(dx * dx + dy * dy + dz * dz)
            if (jump > gateM) { priorReject = true; reason = "consensus jumped %.0f m > gate".format(jump) }
        }
        return RansacResult(true, finalFix, inlierIds.toList(), excluded, residuals, thresholds,
            best.size, priorReject, reason)
    }

    // --- helpers ---
    private fun randomSubset(rng: java.util.Random, n: Int, k: Int): IntArray {
        val chosen = LinkedHashSet<Int>()
        while (chosen.size < k) chosen.add(rng.nextInt(n))
        return chosen.toIntArray()
    }

    private fun allCombinations(n: Int, k: Int): List<IntArray> {
        val out = ArrayList<IntArray>()
        val idx = IntArray(k) { it }
        while (true) {
            out.add(idx.copyOf())
            var i = k - 1
            while (i >= 0 && idx[i] == n - k + i) i--
            if (i < 0) break
            idx[i]++
            for (j in i + 1 until k) idx[j] = idx[j - 1] + 1
        }
        return out
    }
}
