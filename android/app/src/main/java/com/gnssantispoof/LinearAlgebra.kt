package com.gnssantispoof

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/**
 * Pure-Kotlin linear algebra + chi-square helpers — twins of the linear-algebra
 * section of `src/core/solver.py` and `src/antispoof/stats.py`. No external math libs.
 */
object LinAlg {

    /** Solve (Hᵀ W H) dx = Hᵀ W r. */
    fun solveWls(h: Array<DoubleArray>, r: DoubleArray, w: DoubleArray): DoubleArray {
        val nUnk = h[0].size
        val hth = Array(nUnk) { DoubleArray(nUnk) }
        val htr = DoubleArray(nUnk)
        for (i in h.indices) {
            val wi = w[i]; val hi = h[i]; val ri = r[i]
            for (j in 0 until nUnk) {
                htr[j] += hi[j] * wi * ri
                val hijw = hi[j] * wi
                for (k in 0 until nUnk) hth[j][k] += hijw * hi[k]
            }
        }
        return solve(hth, htr)
    }

    /** Gaussian elimination with partial pivoting. */
    fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val n = a.size
        val m = Array(n) { i -> a[i].copyOf(n + 1).also { it[n] = b[i] } }
        for (i in 0 until n) {
            var piv = i
            for (k in i + 1 until n) if (abs(m[k][i]) > abs(m[piv][i])) piv = k
            if (piv != i) { val t = m[i]; m[i] = m[piv]; m[piv] = t }
            if (abs(m[i][i]) < 1e-12) throw ArithmeticException("singular")
            for (k in i + 1 until n) {
                val f = m[k][i] / m[i][i]
                for (j in i..n) m[k][j] -= f * m[i][j]
            }
        }
        val x = DoubleArray(n)
        for (i in n - 1 downTo 0) {
            var s = m[i][n]
            for (j in i + 1 until n) s -= m[i][j] * x[j]
            x[i] = s / m[i][i]
        }
        return x
    }

    fun invert(a: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size
        val m = Array(n) { i -> DoubleArray(2 * n).also { row ->
            a[i].copyInto(row, 0); row[n + i] = 1.0
        } }
        for (i in 0 until n) {
            var piv = i
            for (k in i + 1 until n) if (abs(m[k][i]) > abs(m[piv][i])) piv = k
            if (piv != i) { val t = m[i]; m[i] = m[piv]; m[piv] = t }
            if (abs(m[i][i]) < 1e-12) throw ArithmeticException("singular")
            val inv = 1.0 / m[i][i]
            for (j in 0 until 2 * n) m[i][j] *= inv
            for (k in 0 until n) if (k != i) {
                val f = m[k][i]
                for (j in 0 until 2 * n) m[k][j] -= f * m[i][j]
            }
        }
        return Array(n) { i -> DoubleArray(n) { j -> m[i][j + n] } }
    }
}

/** Chi-square survival / inverse-survival — twin of `src/antispoof/stats.py`. */
object Chi2 {
    private const val EPS = 1e-12
    private const val FPMIN = 1e-300

    private fun gammpSeries(a: Double, x: Double): Double {
        if (x <= 0.0) return 0.0
        var ap = a; var sum = 1.0 / a; var del = sum
        repeat(300) {
            ap += 1.0; del *= x / ap; sum += del
            if (abs(del) < abs(sum) * EPS) return@repeat
        }
        return sum * exp(-x + a * ln(x) - lgamma(a))
    }

    private fun gammqCf(a: Double, x: Double): Double {
        var b = x + 1.0 - a; var c = 1.0 / FPMIN; var d = 1.0 / b; var h = d
        for (i in 1 until 300) {
            val an = -i * (i - a)
            b += 2.0; d = an * d + b; if (abs(d) < FPMIN) d = FPMIN
            c = b + an / c; if (abs(c) < FPMIN) c = FPMIN
            d = 1.0 / d; val del = d * c; h *= del
            if (abs(del - 1.0) < EPS) break
        }
        return exp(-x + a * ln(x) - lgamma(a)) * h
    }

    private fun gammp(a: Double, x: Double): Double =
        if (x < a + 1.0) gammpSeries(a, x) else 1.0 - gammqCf(a, x)

    fun sf(x: Double, dof: Int): Double = if (x <= 0.0) 1.0 else 1.0 - gammp(dof / 2.0, x / 2.0)

    fun isf(pfa: Double, dof: Int): Double {
        if (pfa <= 0.0) return Double.POSITIVE_INFINITY
        if (pfa >= 1.0) return 0.0
        var lo = 0.0; var hi = maxOf(10.0, dof + 10.0)
        while (sf(hi, dof) > pfa && hi < 1e7) hi *= 2.0
        repeat(200) {
            val mid = 0.5 * (lo + hi)
            if (sf(mid, dof) > pfa) lo = mid else hi = mid
        }
        return 0.5 * (lo + hi)
    }

    // Lanczos log-gamma.
    private fun lgamma(x: Double): Double {
        val g = doubleArrayOf(
            676.5203681218851, -1259.1392167224028, 771.32342877765313,
            -176.61502916214059, 12.507343278686905, -0.13857109526572012,
            9.9843695780195716e-6, 1.5056327351493116e-7
        )
        if (x < 0.5) return ln(Math.PI / kotlin.math.sin(Math.PI * x)) - lgamma(1.0 - x)
        val z = x - 1.0
        var a = 0.99999999999980993
        val t = z + 7.5
        for (i in g.indices) a += g[i] / (z + i + 1)
        return 0.5 * ln(2 * Math.PI) + (z + 0.5) * ln(t) - t + ln(a)
    }
}
