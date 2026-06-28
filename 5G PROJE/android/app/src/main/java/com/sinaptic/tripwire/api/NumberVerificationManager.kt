package com.sinaptic.tripwire.api

import android.content.Context
import android.content.Intent
import android.net.Uri
import com.sinaptic.tripwire.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import net.openid.appauth.AuthState
import net.openid.appauth.AuthorizationRequest
import net.openid.appauth.AuthorizationResponse
import net.openid.appauth.AuthorizationService
import net.openid.appauth.AuthorizationServiceConfiguration
import net.openid.appauth.ResponseTypeValues
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

/** OIDC Authorization Code + PKCE coordinator for user-bound Number Verification. */
class NumberVerificationManager(context: Context) {
    private val service = AuthorizationService(context)
    private val http = OkHttpClient()
    @Volatile private var numberVerificationToken: String? = null
    @Volatile private var verifiedPhoneNumber: String? = null
    @Volatile private var localAppSessionToken: String? = null

    fun authorizationIntent(): Intent {
        require(BuildConfig.OIDC_AUTHORIZATION_ENDPOINT.startsWith("https://")) { "HTTPS OIDC authorization endpoint missing" }
        require(BuildConfig.OIDC_TOKEN_ENDPOINT.startsWith("https://")) { "HTTPS OIDC token endpoint missing" }
        require(BuildConfig.OIDC_CLIENT_ID.isNotBlank()) { "OIDC public client id missing" }
        require(BuildConfig.API_BASE_URL.startsWith("https://")) { "HTTPS team BFF endpoint missing" }
        val configuration = AuthorizationServiceConfiguration(
            Uri.parse(BuildConfig.OIDC_AUTHORIZATION_ENDPOINT),
            Uri.parse(BuildConfig.OIDC_TOKEN_ENDPOINT),
        )
        val request = AuthorizationRequest.Builder(
            configuration,
            BuildConfig.OIDC_CLIENT_ID,
            ResponseTypeValues.CODE,
            Uri.parse(BuildConfig.OIDC_REDIRECT_URI),
        )
            .setScope("openid number-verification:verify")
            .setPrompt("none")
            .build()
        // AppAuth generates and verifies PKCE code_verifier/code_challenge by default.
        return service.getAuthorizationRequestIntent(request)
    }

    fun handleAuthorizationResult(intent: Intent, onComplete: (Boolean) -> Unit) {
        val response = AuthorizationResponse.fromIntent(intent)
        val error = net.openid.appauth.AuthorizationException.fromIntent(intent)
        if (response == null || error != null) {
            onComplete(false)
            return
        }
        val state = AuthState(response, error)
        service.performTokenRequest(response.createTokenExchangeRequest()) { token, tokenError ->
            state.update(token, tokenError)
            val accessToken = token?.accessToken
            val expiresAt = token?.accessTokenExpirationTime ?: 0L
            // Number Verification token is accepted only when short-lived (<=300 s).
            val remainingMs = expiresAt - System.currentTimeMillis()
            val shortLived = accessToken != null && remainingMs in 1..300_000L
            if (shortLived) {
                numberVerificationToken = accessToken
            }
            onComplete(shortLived)
        }
    }

    fun accessToken(): String? = numberVerificationToken

    suspend fun verify(phoneNumber: String, deviceId: String): Boolean = withContext(Dispatchers.IO) {
        require(Regex("^\\+[1-9][0-9]{7,14}$").matches(phoneNumber)) { "E.164 phone number required" }
        val token = accessToken() ?: return@withContext false
        val body = JSONObject()
            .put("phone_number", phoneNumber)
            .put("device_id", deviceId)
            .toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType())
        val request = Request.Builder()
            .url("${BuildConfig.API_BASE_URL.trimEnd('/')}/auth/verify")
            .header("Authorization", "Bearer $token")
            .post(body)
            .build()
        try {
            http.newCall(request).execute().use { response ->
                val payload = JSONObject(response.body?.string().orEmpty())
                val verified = response.isSuccessful && payload.optBoolean("verified", false)
                if (verified) {
                    verifiedPhoneNumber = phoneNumber
                    localAppSessionToken = payload.getString("app_session_token")
                }
                verified
            }
        } catch (_: Exception) {
            false
        } finally {
            numberVerificationToken = null
        }
    }

    fun verifiedPhone(): String? = verifiedPhoneNumber
    fun appSessionToken(): String? = localAppSessionToken

    fun close() {
        numberVerificationToken = null
        localAppSessionToken = null
        verifiedPhoneNumber = null
        http.dispatcher.cancelAll()
        service.dispose()
    }
}
