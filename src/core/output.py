"""Write the per-epoch fix track to CSV and KML."""

import csv
import math
from datetime import timezone


CSV_FIELDS = [
    "utc_time",
    "gps_week", "gps_sow",
    "latitude_deg", "longitude_deg", "altitude_m",
    "ecef_x", "ecef_y", "ecef_z",
    "vx_mps", "vy_mps", "vz_mps", "speed_mps",
    "clock_bias_m", "clock_drift_m_s",
    "n_sats", "pdop", "hdop", "vdop",
]


def write_csv(path, fixes):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for f in fixes:
            row = {k: f.get(k, "") for k in CSV_FIELDS}
            if "utc_time" in f and hasattr(f["utc_time"], "isoformat"):
                row["utc_time"] = f["utc_time"].astimezone(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ")
            writer.writerow(row)


def write_kml(path, fixes, name="GNSS Ex0 track"):
    coords = []
    for f in fixes:
        if f.get("latitude_deg") is None:
            continue
        coords.append(f"{f['longitude_deg']:.8f},{f['latitude_deg']:.8f},{f['altitude_m']:.2f}")
    if not coords:
        return
    coords_text = "\n          ".join(coords)
    # Place a few sample markers (start/end/mid) for orientation.
    markers = []
    for label, idx in (("Start", 0), ("End", len(fixes) - 1)):
        f = fixes[idx]
        if f.get("latitude_deg") is None:
            continue
        markers.append(
            f"    <Placemark>\n"
            f"      <name>{label}</name>\n"
            f"      <Point><coordinates>"
            f"{f['longitude_deg']:.8f},{f['latitude_deg']:.8f},{f['altitude_m']:.2f}"
            f"</coordinates></Point>\n"
            f"    </Placemark>"
        )
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <Style id="track">
      <LineStyle><color>ff0066ff</color><width>3</width></LineStyle>
    </Style>
    <Placemark>
      <name>Path</name>
      <styleUrl>#track</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>
          {coords_text}
        </coordinates>
      </LineString>
    </Placemark>
{chr(10).join(markers)}
  </Document>
</kml>
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(kml)
