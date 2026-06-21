"""Pure-Python chi-square helpers for RAIM (no SciPy dependency).

RAIM compares a weighted sum-of-squared-residuals test statistic against a
chi-square threshold T = chi2.isf(Pfa, dof). To keep the whole project free of
third-party dependencies we implement the regularised incomplete gamma function
(Numerical Recipes style) and derive the chi-square survival function and its
inverse by bisection.

Validated against the standard table, e.g. chi2.isf(1e-5, dof=4) = 28.473.
"""

from __future__ import annotations

import math

_MAXIT = 300
_EPS = 1e-12
_FPMIN = 1e-300


def _gammp_series(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) via series (x < a+1)."""
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_MAXIT):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammq_cf(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) via continued fraction (x >= a+1)."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAXIT):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammp(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("invalid arguments to gammp")
    if x < a + 1.0:
        return _gammp_series(a, x)
    return 1.0 - _gammq_cf(a, x)


def chi2_cdf(x: float, dof: int) -> float:
    """Chi-square cumulative distribution function."""
    if x <= 0.0:
        return 0.0
    return gammp(dof / 2.0, x / 2.0)


def chi2_sf(x: float, dof: int) -> float:
    """Chi-square survival function 1 - CDF."""
    return 1.0 - chi2_cdf(x, dof)


def chi2_isf(pfa: float, dof: int) -> float:
    """Inverse survival function: return x with P(X > x) = pfa for X ~ chi2(dof)."""
    if pfa <= 0.0:
        return float("inf")
    if pfa >= 1.0:
        return 0.0
    # Bisection on the survival function (monotonically decreasing in x).
    lo, hi = 0.0, max(10.0, dof + 10.0)
    # Expand the upper bound until sf(hi) < pfa.
    while chi2_sf(hi, dof) > pfa:
        hi *= 2.0
        if hi > 1e7:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_sf(mid, dof) > pfa:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return 0.5 * (lo + hi)
