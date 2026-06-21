package com.gnssantispoof

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Residual-based RAIM fault detection & exclusion — twin of `src/antispoof/raim.py`.
 * SSE = wᵀWw tested against χ²₍₁₋Pfa₎(n−m); FDE removes the largest normalised residual.
 */
data class RaimResult(
    val ok: Boolean, val faultDetected: Boolean = false,
    val sse: Double = 0.0, val threshold: Double = 0.0, val dof: Int = 0,
    val excludedIds: List<String> = emptyList(),
)

object Raim {
    private const val SIGMA_RHO = 5.0
    private const val PFA = 1e-5

    private fun sigma(s: SatState): Double {
        val base = if (s.prUncertaintyM != null && s.prUncertaintyM > 0) s.prUncertaintyM else SIGMA_RHO
        return if (s.elevationRad != null) base / max(sin(s.elevationRad), 0.1) else base
    }

    private data class Sol(val wRes: DoubleArray, val weights: DoubleArray,
                           val sse: Double, val dof: Int, val sii: DoubleArray)

    private fun solveResiduals(states: List<SatState>): Sol? {
        val active = Solver.activeSys(states)
        val m = 4 + active.size
        val n = states.size
        if (n <= m) return null
        // Seed near Earth surface from the first SV sub-point.
        val s0 = states[0]
        val nrm = sqrt(s0.x * s0.x + s0.y * s0.y + s0.z * s0.z)
        val rx = doubleArrayOf(s0.x / nrm * 6371000.0, s0.y / nrm * 6371000.0, s0.z / nrm * 6371000.0)
        var clk = 0.0
        val isb = HashMap<String, Double>().apply { active.forEach { put(it, 0.0) } }

        fun build(): Triple<Array<DoubleArray>, DoubleArray, DoubleArray> {
            val h = Array(n) { DoubleArray(m) }; val z = DoubleArray(n); val w = DoubleArray(n)
            for (i in states.indices) {
                val s = states[i]
                val dx = s.x - rx[0]; val dy = s.y - rx[1]; val dz = s.z - rx[2]
                val rho = sqrt(dx * dx + dy * dy + dz * dz)
                val extra = if (s.sys != "G") isb[s.sys] ?: 0.0 else 0.0
                h[i][0] = -dx / rho; h[i][1] = -dy / rho; h[i][2] = -dz / rho; h[i][3] = 1.0
                if (s.sys in active) h[i][4 + active.indexOf(s.sys)] = 1.0
                z[i] = s.pseudorangeM - (rho + clk + extra - Solver.C * s.svClockS)
                val sg = sigma(s); w[i] = 1.0 / (sg * sg)
            }
            return Triple(h, z, w)
        }

        repeat(15) {
            val (h, z, w) = build()
            val d = try { LinAlg.solveWls(h, z, w) } catch (e: ArithmeticException) { return null }
            rx[0] += d[0]; rx[1] += d[1]; rx[2] += d[2]; clk += d[3]
            active.forEachIndexed { i, sy -> isb[sy] = (isb[sy] ?: 0.0) + d[4 + i] }
            if (abs(d[0]) < 1e-3 && abs(d[1]) < 1e-3 && abs(d[2]) < 1e-3) return@repeat
        }
        val (h, z, w) = build()
        val d = try { LinAlg.solveWls(h, z, w) } catch (e: ArithmeticException) { return null }
        val wRes = DoubleArray(n) { i -> z[i] - (0 until m).sumOf { j -> h[i][j] * d[j] } }
        val sse = (0 until n).sumOf { w[it] * wRes[it] * wRes[it] }
        // Hat diagonal for normalised residuals.
        val htwh = Array(m) { DoubleArray(m) }
        for (i in 0 until n) for (a in 0 until m) for (b in 0 until m) htwh[a][b] += h[i][a] * w[i] * h[i][b]
        val inv = try { LinAlg.invert(htwh) } catch (e: ArithmeticException) { return null }
        val sii = DoubleArray(n) { i ->
            var quad = 0.0
            for (a in 0 until m) { var ma = 0.0; for (b in 0 until m) ma += inv[a][b] * h[i][b]; quad += h[i][a] * ma }
            max(1.0 - w[i] * quad, 1e-6)
        }
        return Sol(wRes, w, sse, n - m, sii)
    }

    fun fde(states: List<SatState>, maxExclude: Int = 4): RaimResult {
        var working = states.toList()
        val excluded = ArrayList<String>()
        var sol = solveResiduals(working) ?: return RaimResult(false)
        while (true) {
            val dof = sol.dof
            val t = if (dof >= 1) Chi2.isf(PFA, dof) else Double.POSITIVE_INFINITY
            if (sol.sse <= t || dof < 1)
                return RaimResult(true, excluded.isNotEmpty(), sol.sse, t, dof, excluded)
            if (working.size - 1 < (4 + Solver.activeSys(working).size) + 1 || excluded.size >= maxExclude)
                return RaimResult(true, true, sol.sse, t, dof, excluded)
            var worst = 0; var worstVal = -1.0
            for (i in working.indices) {
                val nr = abs(sol.wRes[i]) * sqrt(sol.weights[i]) / sqrt(sol.sii[i])
                if (nr > worstVal) { worstVal = nr; worst = i }
            }
            excluded.add(working[worst].satId)
            working = working.filterIndexed { i, _ -> i != worst }
            sol = solveResiduals(working) ?: return RaimResult(true, true, sol.sse, t, dof, excluded)
        }
    }

    fun consistency(states: List<SatState>): Boolean {
        val sol = solveResiduals(states) ?: return true
        val t = if (sol.dof >= 1) Chi2.isf(PFA, sol.dof) else Double.POSITIVE_INFINITY
        return sol.sse <= t
    }
}
