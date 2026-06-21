"""Write engine outputs to CSV / KML and print a run summary."""

from __future__ import annotations

import csv
from datetime import timezone
from typing import List

from ..antispoof.detector import SpoofStatus
from .engine import EngineOutput

CSV_FIELDS = [
    "utc_time", "status", "confidence", "n_sats", "spoofed_prns",
    "latitude_deg", "longitude_deg", "altitude_m",
    "speed_mps", "pdop", "hdop",
    "naive_latitude_deg", "naive_longitude_deg",
    "rf_verdict", "agc_drop_db", "raim_sse", "raim_threshold", "dead_reckoned",
]


def write_engine_csv(path: str, outputs: List[EngineOutput]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for o in outputs:
            w.writerow({
                "utc_time": o.utc_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "status": o.status.value, "confidence": o.confidence, "n_sats": o.n_sats,
                "spoofed_prns": "|".join(o.spoofed_prns),
                "latitude_deg": _f(o.lat_deg), "longitude_deg": _f(o.lon_deg),
                "altitude_m": _f(o.alt_m), "speed_mps": _f(o.speed_mps),
                "pdop": _f(o.pdop), "hdop": _f(o.hdop),
                "naive_latitude_deg": _f(o.naive_lat_deg), "naive_longitude_deg": _f(o.naive_lon_deg),
                "rf_verdict": o.rf_verdict or "", "agc_drop_db": _f(o.agc_drop_db),
                "raim_sse": _f(o.raim_sse), "raim_threshold": _f(o.raim_threshold),
                "dead_reckoned": int(o.dead_reckoned),
            })


def write_spoof_events(path: str, outputs: List[EngineOutput]) -> int:
    """Write only the epochs flagged SUSPECT/SPOOFED. Returns the count."""
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utc_time", "status", "confidence", "spoofed_prns", "rf_verdict", "reasons"])
        for o in outputs:
            if o.status == SpoofStatus.CLEAN:
                continue
            n += 1
            w.writerow([
                o.utc_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                o.status.value, o.confidence, "|".join(o.spoofed_prns),
                o.rf_verdict or "", "; ".join(o.reasons),
            ])
    return n


def write_engine_kml(path: str, outputs: List[EngineOutput], name: str = "GNSS-AntiSpoof track") -> None:
    """Two line strings: trusted (mitigated) track and naive track, plus spoof markers."""
    trusted = [(o.lon_deg, o.lat_deg, o.alt_m or 0.0) for o in outputs if o.lat_deg is not None]
    naive = [(o.naive_lon_deg, o.naive_lat_deg, 0.0) for o in outputs if o.naive_lat_deg is not None]

    def coords(pts):
        return "\n          ".join(f"{lon:.8f},{lat:.8f},{alt:.2f}" for lon, lat, alt in pts)

    markers = []
    for o in outputs:
        if o.status == SpoofStatus.SPOOFED and o.lat_deg is not None:
            markers.append(
                f"    <Placemark><name>SPOOFED {'|'.join(o.spoofed_prns)}</name>"
                f"<Point><coordinates>{o.lon_deg:.8f},{o.lat_deg:.8f},{o.alt_m or 0:.1f}</coordinates></Point>"
                f"</Placemark>")
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <Style id="trusted"><LineStyle><color>ff00aa00</color><width>4</width></LineStyle></Style>
    <Style id="naive"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle></Style>
    <Placemark><name>Trusted (mitigated)</name><styleUrl>#trusted</styleUrl>
      <LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode>
        <coordinates>
          {coords(trusted)}
        </coordinates></LineString></Placemark>
    <Placemark><name>Naive (unprotected)</name><styleUrl>#naive</styleUrl>
      <LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode>
        <coordinates>
          {coords(naive)}
        </coordinates></LineString></Placemark>
{chr(10).join(markers[:200])}
  </Document>
</kml>
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(kml)


def print_summary(outputs: List[EngineOutput]) -> None:
    n = len(outputs)
    clean = sum(1 for o in outputs if o.status == SpoofStatus.CLEAN)
    suspect = sum(1 for o in outputs if o.status == SpoofStatus.SUSPECT)
    spoofed = sum(1 for o in outputs if o.status == SpoofStatus.SPOOFED)
    dr = sum(1 for o in outputs if o.dead_reckoned)
    prns = {}
    for o in outputs:
        for p in o.spoofed_prns:
            prns[p] = prns.get(p, 0) + 1
    print(f"epochs: {n}   CLEAN {clean}   SUSPECT {suspect}   SPOOFED {spoofed}   (dead-reckoned {dr})")
    if prns:
        top = sorted(prns.items(), key=lambda kv: -kv[1])
        print("flagged satellites (epochs): " + ", ".join(f"{p}:{c}" for p, c in top))


def _f(v):
    return "" if v is None else f"{v:.6f}" if abs(v) < 1e6 else f"{v:.3f}"
