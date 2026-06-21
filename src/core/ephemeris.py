"""Compute satellite ECEF position, velocity and clock correction from broadcast ephemeris.

References:
  - IS-GPS-200 (GPS broadcast Keplerian propagator)
  - Galileo OS SIS ICD (same Keplerian formulation as GPS)
  - BeiDou ICD-B1I (same formulation; toe is BDT seconds-of-week)
  - GLONASS ICD (numerical integration of ECEF state with lunisolar accel)
"""

import math
from typing import Tuple

# Universal gravitational parameter (m^3 / s^2)
MU_GPS = 3.986005e14            # GPS / Galileo
MU_BDS = 3.986004418e14         # BeiDou
MU_GLO = 3.9860044e14           # GLONASS

# Earth rotation rate (rad/s)
OMEGA_E = 7.2921151467e-5
OMEGA_E_BDS = 7.292115e-5
OMEGA_E_GLO = 7.292115e-5

# Speed of light (m/s)
C = 299_792_458.0

F_REL = -4.442807633e-10  # relativistic correction constant (s/sqrt(m))


def _solve_kepler(M, e, tol=1e-12, max_iter=30):
    """Iteratively solve Kepler's equation M = E - e sin(E) for E."""
    E = M
    for _ in range(max_iter):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


def sv_clock_correction(eph, t_gps_sow):
    """Compute satellite clock bias (seconds) at transmit time t_gps_sow (seconds of week
    in the broadcast frame's time system)."""
    dt = t_gps_sow - eph.toe
    if dt > 302400:
        dt -= 604800
    elif dt < -302400:
        dt += 604800
    # relativistic term filled in later by caller (needs E)
    return eph.af0 + eph.af1 * dt + eph.af2 * dt * dt


def compute_sv_ecef(eph, t_transmit_sow):
    """Compute satellite ECEF position (m) and velocity (m/s) at transmit time.

    `t_transmit_sow` is seconds-of-week in the *broadcast* system's time scale
    (already corrected by the caller for system-time offset and the sat clock).

    Returns (x, y, z, vx, vy, vz, sv_clock_relativistic_correction_seconds).
    """
    sys = eph.sys
    if sys == "C":
        mu, omega_e = MU_BDS, OMEGA_E_BDS
    else:
        mu, omega_e = MU_GPS, OMEGA_E

    A = eph.sqrt_a ** 2
    n0 = math.sqrt(mu / (A ** 3))
    tk = t_transmit_sow - eph.toe
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800

    n = n0 + eph.delta_n
    M = eph.m0 + n * tk
    E = _solve_kepler(M, eph.e)

    sinE = math.sin(E); cosE = math.cos(E)
    sqrt_1me2 = math.sqrt(1.0 - eph.e * eph.e)

    # True anomaly
    sin_nu = sqrt_1me2 * sinE / (1.0 - eph.e * cosE)
    cos_nu = (cosE - eph.e) / (1.0 - eph.e * cosE)
    nu = math.atan2(sin_nu, cos_nu)

    phi = nu + eph.omega
    sin2phi = math.sin(2 * phi); cos2phi = math.cos(2 * phi)
    du = eph.cuc * cos2phi + eph.cus * sin2phi
    dr = eph.crc * cos2phi + eph.crs * sin2phi
    di = eph.cic * cos2phi + eph.cis * sin2phi

    u = phi + du
    r = A * (1.0 - eph.e * cosE) + dr
    i = eph.i0 + di + eph.idot * tk

    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)

    # BeiDou GEO satellites (PRN 1-5, 59-63) use a different rotation; here we treat all BDS
    # via the standard MEO/IGSO formula. Phones rarely track BDS GEOs, so this is acceptable.
    omega_k = (eph.omega0 + (eph.omega_dot - omega_e) * tk
               - omega_e * eph.toe)
    sin_omega = math.sin(omega_k); cos_omega = math.cos(omega_k)
    sini = math.sin(i); cosi = math.cos(i)

    x = x_orb * cos_omega - y_orb * cosi * sin_omega
    y = x_orb * sin_omega + y_orb * cosi * cos_omega
    z = y_orb * sini

    # Velocity (analytic derivative)
    Edot = n / (1.0 - eph.e * cosE)
    nu_dot = sqrt_1me2 * Edot / (1.0 - eph.e * cosE)
    udot = nu_dot + 2.0 * (eph.cus * cos2phi - eph.cuc * sin2phi) * nu_dot
    rdot = A * eph.e * sinE * Edot + 2.0 * (eph.crs * cos2phi - eph.crc * sin2phi) * nu_dot
    idot = eph.idot + 2.0 * (eph.cis * cos2phi - eph.cic * sin2phi) * nu_dot

    xdot_orb = rdot * math.cos(u) - r * math.sin(u) * udot
    ydot_orb = rdot * math.sin(u) + r * math.cos(u) * udot
    omega_dot_k = eph.omega_dot - omega_e

    vx = (xdot_orb * cos_omega
          - ydot_orb * cosi * sin_omega
          + y_orb * sini * sin_omega * idot
          - (x_orb * sin_omega + y_orb * cosi * cos_omega) * omega_dot_k)
    vy = (xdot_orb * sin_omega
          + ydot_orb * cosi * cos_omega
          - y_orb * sini * cos_omega * idot
          + (x_orb * cos_omega - y_orb * cosi * sin_omega) * omega_dot_k)
    vz = ydot_orb * sini + y_orb * cosi * idot

    rel = F_REL * eph.e * eph.sqrt_a * sinE  # seconds
    return x, y, z, vx, vy, vz, rel


def apply_earth_rotation(x, y, z, transit_time, omega_e=OMEGA_E):
    """Rotate sat ECEF backward to compensate for Earth rotation during signal transit."""
    theta = omega_e * transit_time
    cos_t = math.cos(theta); sin_t = math.sin(theta)
    return (cos_t * x + sin_t * y,
            -sin_t * x + cos_t * y,
            z)


# ---------------- GLONASS ----------------

def _glonass_accel(state, n):
    """Right-hand side of GLONASS PZ-90 ODE. state = [x, y, z, vx, vy, vz]."""
    x, y, z, vx, vy, vz = state
    a_e = 6378136.0
    mu = MU_GLO
    j02 = 1.0826257e-3
    omega = OMEGA_E_GLO
    r = math.sqrt(x * x + y * y + z * z)
    rho = a_e / r
    xb = x / r; yb = y / r; zb = z / r
    fac = 1.5 * j02 * mu * (a_e ** 2) / (r ** 4)
    ax = -mu * xb / (r * r) + fac * xb * (1.0 - 5.0 * zb * zb) + omega * omega * x + 2.0 * omega * vy
    ay = -mu * yb / (r * r) + fac * yb * (1.0 - 5.0 * zb * zb) + omega * omega * y - 2.0 * omega * vx
    az = -mu * zb / (r * r) + fac * zb * (3.0 - 5.0 * zb * zb)
    return (vx, vy, vz, ax + n[0], ay + n[1], az + n[2])


def _rk4_step(state, n, dt):
    k1 = _glonass_accel(state, n)
    s2 = tuple(state[i] + 0.5 * dt * k1[i] for i in range(6))
    k2 = _glonass_accel(s2, n)
    s3 = tuple(state[i] + 0.5 * dt * k2[i] for i in range(6))
    k3 = _glonass_accel(s3, n)
    s4 = tuple(state[i] + dt * k3[i] for i in range(6))
    k4 = _glonass_accel(s4, n)
    return tuple(state[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0 for i in range(6))


def propagate_glonass(eph, dt_seconds: float) -> Tuple[float, ...]:
    """Integrate GLONASS state from toc to toc + dt_seconds using RK4. PZ-90 km/s units in
    the broadcast record are converted to m/s here."""
    state = (eph.x * 1000.0, eph.y * 1000.0, eph.z * 1000.0,
             eph.vx * 1000.0, eph.vy * 1000.0, eph.vz * 1000.0)
    accel = (eph.ax * 1000.0, eph.ay * 1000.0, eph.az * 1000.0)
    step = 30.0 if dt_seconds >= 0 else -30.0
    remaining = dt_seconds
    while abs(remaining) > 1e-6:
        h = step if abs(remaining) > abs(step) else remaining
        state = _rk4_step(state, accel, h)
        remaining -= h
    return state  # (x,y,z,vx,vy,vz)
