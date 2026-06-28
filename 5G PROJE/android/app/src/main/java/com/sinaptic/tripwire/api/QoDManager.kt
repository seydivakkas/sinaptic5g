package com.sinaptic.tripwire.api

import android.util.Log
import com.sinaptic.tripwire.BuildConfig
import com.sinaptic.tripwire.model.AnalysisResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Android talks only to the team BFF. Provider credentials and provider-specific
 * QoD fields never enter the APK.
 */
class QoDManager(@Suppress("UNUSED_PARAMETER") context: android.content.Context) {
    companion object {
        private const val TAG = "QoDManager"
        private val JSON = "application/json; charset=utf-8".toMediaType()
    }

    @Volatile private var activeSessionId: String? = null
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    suspend fun startSession(
        phoneNumber: String,
        signals: AnalysisResult,
        accessToken: String,
        deviceId: String,
    ): String? = withContext(Dispatchers.IO) {
        if (!signals.targetPresent || !signals.vehicleApproaching) return@withContext null
        if (activeSessionId != null) return@withContext null
        if (!BuildConfig.API_BASE_URL.startsWith("https://")) {
            Log.w(TAG, "HTTPS BFF endpoint missing; Best Effort continues")
            return@withContext null
        }

        val body = JSONObject().apply {
            put("device_id", deviceId)
            put("phone_number", phoneNumber)
            put("target_present", signals.targetPresent)
            put("vehicle_is_approaching", signals.vehicleApproaching)
            put("recognizability_gap", signals.recognizabilityGap)
            put("model_uncertainty", signals.modelUncertainty)
            put("media_degradation", signals.mediaDegradation)
            put("network_degradation", signals.networkDegradation)
        }.toString().toRequestBody(JSON)

        val request = Request.Builder()
            .url("${BuildConfig.API_BASE_URL.trimEnd('/')}/qod/start")
            .header("Authorization", "Bearer $accessToken")
            .post(body)
            .build()
        try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.w(TAG, "QoD BFF rejected request: ${response.code}; Best Effort continues")
                    return@withContext null
                }
                val result = JSONObject(response.body?.string().orEmpty())
                if (result.optString("status") != "active") {
                    Log.i(TAG, "QoD not requested: ${result.optString("reason", "benefit_gate")}")
                    return@withContext null
                }
                val sessionId = result.getString("session_id")
                activeSessionId = sessionId
                sessionId
            }
        } catch (error: Exception) {
            Log.w(TAG, "QoD unavailable; Best Effort continues", error)
            null
        }
    }

    suspend fun stopSession(sessionId: String, deviceId: String, accessToken: String): Boolean =
        withContext(Dispatchers.IO) {
            if (!BuildConfig.API_BASE_URL.startsWith("https://")) return@withContext false
            val request = Request.Builder()
                .url("${BuildConfig.API_BASE_URL.trimEnd('/')}/qod/$sessionId?device_id=$deviceId")
                .header("Authorization", "Bearer $accessToken")
                .delete()
                .build()
            try {
                client.newCall(request).execute().use { response ->
                    if (response.isSuccessful) activeSessionId = null
                    response.isSuccessful
                }
            } catch (error: Exception) {
                Log.w(TAG, "QoD stop failed", error)
                false
            }
        }

    fun cleanup() {
        activeSessionId = null
        client.dispatcher.cancelAll()
    }
}
