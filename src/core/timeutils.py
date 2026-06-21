"""Time-system helpers (calendar <-> GPS week/sec <-> UTC)."""

from datetime import datetime, timedelta, timezone

GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
SECONDS_PER_WEEK = 7 * 24 * 3600

# Constellation time-system offsets vs GPST (seconds).
# Galileo time is aligned with GPST (sub-microsecond).
# BeiDou time is GPST - 14 s (different epoch and leap-second history).
# GLONASS uses UTC(SU)+3h — handled separately.
BDS_GPS_OFFSET = 14.0


def utc_to_gps_week_sec(utc_dt: datetime, leap_seconds: int = 18):
    """UTC datetime -> (gps_week, gps_sec_of_week). leap_seconds = GPST-UTC (18 since 2017)."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    gps_dt = utc_dt + timedelta(seconds=leap_seconds)
    delta = gps_dt - GPS_EPOCH
    total = delta.total_seconds()
    week = int(total // SECONDS_PER_WEEK)
    sow = total - week * SECONDS_PER_WEEK
    return week, sow


def gps_week_sec_to_utc(week: int, sow: float, leap_seconds: int = 18):
    """Inverse of utc_to_gps_week_sec."""
    gps_dt = GPS_EPOCH + timedelta(seconds=week * SECONDS_PER_WEEK + sow)
    return gps_dt - timedelta(seconds=leap_seconds)


def datetime_to_gps_sow(dt: datetime, leap_seconds: int = 18):
    """Helper used by RINEX parsers: returns GPS sow without leap correction (RINEX timestamps
    are already in GPS time, so leap_seconds=0)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    gps_dt = dt + timedelta(seconds=leap_seconds)
    delta = gps_dt - GPS_EPOCH
    total = delta.total_seconds()
    return total  # absolute GPS seconds since epoch


def gps_seconds_to_week_sow(total_gps_seconds: float):
    week = int(total_gps_seconds // SECONDS_PER_WEEK)
    sow = total_gps_seconds - week * SECONDS_PER_WEEK
    return week, sow
