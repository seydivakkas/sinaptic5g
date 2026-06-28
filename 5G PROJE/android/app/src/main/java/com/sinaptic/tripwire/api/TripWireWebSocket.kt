package com.sinaptic.tripwire.api

import android.util.Log
import com.sinaptic.tripwire.model.AnalysisResult
import com.sinaptic.tripwire.model.BoundingBox
import com.sinaptic.tripwire.model.Detection
import kotlinx.coroutines.*
import okhttp3.*
import org.json.JSONObject

/**
 * TripWireWebSocket — Backend WebSocket İstemcisi
 *
 * Yalnız tespit/telemetri sonuçlarını alır. Medya bu soketten taşınmaz;
 * final sistemde ayrı WebRTC medya düzlemi kullanılır.
 */
class TripWireWebSocket(
    private val serverUrl: String,
    private val deviceId:  String,
    private val onResult:  (String, AnalysisResult) -> Unit,
    private val onConnectionState: (ConnectionState) -> Unit,
) {
    enum class ConnectionState { DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING, FAILED }

    companion object {
        private const val TAG         = "TripWireWS"
        private const val PING_MILLIS = 15_000L
    }

    private val client     = OkHttpClient.Builder()
        .pingInterval(PING_MILLIS, java.util.concurrent.TimeUnit.MILLISECONDS)
        .build()
    private var webSocket: WebSocket? = null
    private var isConnected = false
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var accessToken: String? = null
    private var shouldReconnect = false
    private var reconnectAttempt = 0
    private var reconnectJob: Job? = null

    private val listener = object : WebSocketListener() {
        override fun onOpen(ws: WebSocket, response: Response) {
            isConnected = true
            reconnectAttempt = 0
            onConnectionState(ConnectionState.CONNECTED)
            Log.i(TAG, "WebSocket baglandi: $serverUrl")
            // Kimlik bildirimi
            ws.send(JSONObject().apply {
                put("type",      "hello")
                put("device_id", deviceId)
            }.toString())
        }

        override fun onMessage(ws: WebSocket, text: String) {
            try {
                val json = JSONObject(text)
                if (json.optString("type") in setOf("analysis_result", "analysis_snapshot")) {
                    val result = parseResult(json)
                    onResult(json.getString("event_id"), result)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Mesaj parse hatasi: $e")
            }
        }

        override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
            isConnected = false
            onConnectionState(ConnectionState.FAILED)
            Log.w(TAG, "WebSocket hatasi: ${t.message}")
            scheduleReconnect()
        }

        override fun onClosed(ws: WebSocket, code: Int, reason: String) {
            isConnected = false
            onConnectionState(ConnectionState.DISCONNECTED)
            Log.i(TAG, "WebSocket kapandi: $code $reason")
            scheduleReconnect()
        }
    }

    fun connect(accessToken: String) {
        this.accessToken = accessToken
        shouldReconnect = true
        reconnectJob?.cancel()
        openSocket(accessToken)
    }

    private fun openSocket(accessToken: String) {
        onConnectionState(
            if (reconnectAttempt == 0) ConnectionState.CONNECTING else ConnectionState.RECONNECTING
        )
        val request = Request.Builder()
            .url(serverUrl)
            .addHeader("X-Device-Id", deviceId)
            .addHeader("Authorization", "Bearer $accessToken")
            .build()
        webSocket = client.newWebSocket(request, listener)
    }

    private fun scheduleReconnect() {
        if (!shouldReconnect || reconnectJob?.isActive == true) return
        val token = accessToken ?: return
        reconnectJob = scope.launch {
            reconnectAttempt = (reconnectAttempt + 1).coerceAtMost(5)
            onConnectionState(ConnectionState.RECONNECTING)
            delay((1_000L shl (reconnectAttempt - 1)).coerceAtMost(30_000L))
            if (shouldReconnect && !isConnected) openSocket(token)
        }
    }

    fun disconnect() {
        shouldReconnect = false
        reconnectJob?.cancel()
        webSocket?.close(1000, "Normal kapanış")
        scope.cancel()
        client.dispatcher.executorService.shutdown()
        onConnectionState(ConnectionState.DISCONNECTED)
    }

    private fun parseResult(json: JSONObject): AnalysisResult {
        fun nullableString(name: String): String? =
            json.optString(name).takeIf { it.isNotBlank() && it != "null" }
        val detectionsJson = json.optJSONArray("detections")
        val detections = buildList {
            if (detectionsJson != null) {
                for (index in 0 until detectionsJson.length()) {
                    val item = detectionsJson.getJSONObject(index)
                    val box = item.getJSONObject("bbox")
                    add(Detection(
                        classId = item.optInt("class_id", 5),
                        className = item.optString("class_name", "vehicle"),
                        confidence = item.optDouble("confidence", 0.0).toFloat(),
                        bbox = BoundingBox(
                            box.optDouble("x1", 0.0).toFloat(),
                            box.optDouble("y1", 0.0).toFloat(),
                            box.optDouble("x2", 0.0).toFloat(),
                            box.optDouble("y2", 0.0).toFloat(),
                        ),
                    ))
                }
            }
        }
        return AnalysisResult(
            frameId       = json.optLong("frame_id", 0L),
            riskScore     = json.optDouble("risk_score", 0.0).toFloat(),
            riskLevel     = when (json.optString("risk_level").lowercase()) {
                "kritik" -> com.sinaptic.tripwire.model.RiskLevel.KRITIK
                "orta" -> com.sinaptic.tripwire.model.RiskLevel.ORTA
                else -> com.sinaptic.tripwire.model.RiskLevel.DUSUK
            },
            detections    = detections,
            licensePlate  = nullableString("license_plate"),
            vehicleType   = nullableString("vehicle_type"),
            vehicleColor  = nullableString("vehicle_color"),
            speedKmh      = json.optDouble("speed_kmh", -1.0).takeIf { it >= 0 }?.toFloat(),
            behaviorClass = nullableString("behavior_class"),
            processingMs  = json.optDouble("processing_time_ms", 0.0).toFloat(),
            targetPresent = detections.any { it.className == "vehicle" },
            timestamp     = (json.optDouble("timestamp", System.currentTimeMillis() / 1000.0) * 1000.0).toLong(),
        )
    }
}
