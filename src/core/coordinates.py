"""ECEF <-> geodetic (WGS-84) and ECEF -> ENU conversions."""

import math

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 1.0 - (WGS84_B * WGS84_B) / (WGS84_A * WGS84_A)


def ecef_to_geodetic(x, y, z):
    """Convert ECEF (m) to geodetic (lat rad, lon rad, alt m) — Bowring closed form."""
    a = WGS84_A
    b = WGS84_B
    e2 = WGS84_E2
    ep2 = (a * a - b * b) / (b * b)

    p = math.sqrt(x * x + y * y)
    if p < 1e-9:
        lat = math.copysign(math.pi / 2.0, z)
        lon = 0.0
        alt = abs(z) - b
        return lat, lon, alt

    theta = math.atan2(z * a, p * b)
    lon = math.atan2(y, x)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        p - e2 * a * math.cos(theta) ** 3,
    )
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n
    return lat, lon, alt


def geodetic_to_ecef(lat, lon, alt):
    """lat/lon in radians, alt in meters."""
    a = WGS84_A
    e2 = WGS84_E2
    sl = math.sin(lat)
    cl = math.cos(lat)
    n = a / math.sqrt(1.0 - e2 * sl * sl)
    x = (n + alt) * cl * math.cos(lon)
    y = (n + alt) * cl * math.sin(lon)
    z = (n * (1.0 - e2) + alt) * sl
    return x, y, z


def ecef_to_enu(dx, dy, dz, lat, lon):
    """Rotate ECEF delta into local east-north-up at given lat/lon (rad)."""
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    return e, n, u


def elevation_azimuth(rx_ecef, sv_ecef):
    """Return (elevation rad, azimuth rad) of satellite seen from receiver."""
    lat, lon, _ = ecef_to_geodetic(*rx_ecef)
    dx, dy, dz = sv_ecef[0] - rx_ecef[0], sv_ecef[1] - rx_ecef[1], sv_ecef[2] - rx_ecef[2]
    e, n, u = ecef_to_enu(dx, dy, dz, lat, lon)
    horiz = math.sqrt(e * e + n * n)
    el = math.atan2(u, horiz)
    az = math.atan2(e, n) % (2.0 * math.pi)
    return el, az
