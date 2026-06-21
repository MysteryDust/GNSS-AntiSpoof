package com.gnssantispoof

import android.Manifest
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

/**
 * Minimal demo activity: registers the raw-measurement callback, drives the
 * anti-spoofing engine, and shows the trusted location + spoof status.
 *
 * Wire a concrete [EphemerisProvider] (decode GnssNavigationMessage on-device, or
 * parse a downloaded BRDC file) where indicated below.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var status: TextView
    private lateinit var engine: GnssAntiSpoofEngine

    private val ephemerisProvider = object : EphemerisProvider {
        // TODO: supply broadcast ephemeris. Register a GnssNavigationMessage.Callback to
        // decode subframes on-device, or parse a downloaded RINEX/BRDC nav file.
        override fun ephemerisFor(satId: String, gpsSeconds: Double): KeplerEph? = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this).apply { textSize = 16f; setPadding(32, 64, 32, 32) }
        setContentView(status)

        engine = GnssAntiSpoofEngine(ephemerisProvider)
        engine.output.observe(this) { o ->
            status.text = buildString {
                appendLine("Status: ${o.status} (${o.confidence})")
                appendLine("Satellites: ${o.nSats}   RF: ${o.rfVerdict}")
                if (o.spoofedPrns.isNotEmpty()) appendLine("Spoofed: ${o.spoofedPrns.joinToString()}")
                if (o.latDeg != null) appendLine("Trusted: %.6f, %.6f".format(o.latDeg, o.lonDeg))
                if (o.naiveLatDeg != null) appendLine("Naive:   %.6f, %.6f".format(o.naiveLatDeg, o.naiveLonDeg))
                if (o.deadReckoned) appendLine("(dead-reckoned)")
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), 1)
        } else {
            startGnss()
        }
    }

    override fun onRequestPermissionsResult(rc: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(rc, perms, results)
        if (results.firstOrNull() == PackageManager.PERMISSION_GRANTED) startGnss()
    }

    private fun startGnss() {
        val lm = getSystemService(LOCATION_SERVICE) as LocationManager
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            == PackageManager.PERMISSION_GRANTED) {
            lm.registerGnssMeasurementsCallback(engine.measurementsCallback, Handler(Looper.getMainLooper()))
        }
    }
}
