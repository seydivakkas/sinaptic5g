package com.sinaptic.tripwire.api

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import org.webrtc.*
import java.net.URLEncoder
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

/**
 * Direct Android -> GPU WebRTC publisher. The BFF receives complete SDP only;
 * no encoded frame, RTP packet or camera image is sent to an HTTP endpoint.
 */
class WebRtcMediaClient(
    context: Context,
    private val bffBaseUrl: String,
    private val deviceId: String,
    private val onState: (String) -> Unit,
) {
    companion object {
        private const val TAG = "WebRtcMedia"
        private const val SDP_TIMEOUT_MS = 60_000L
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(25, TimeUnit.SECONDS)
        .build()
    private val factory: PeerConnectionFactory
    private val videoSource: VideoSource
    private val videoTrack: VideoTrack
    private var peer: PeerConnection? = null
    private var started = false
    @Volatile private var framePublishingEnabled = false

    init {
        PeerConnectionFactory.initialize(
            PeerConnectionFactory.InitializationOptions.builder(context.applicationContext)
                .setEnableInternalTracer(false)
                .createInitializationOptions()
        )
        factory = PeerConnectionFactory.builder()
            .setOptions(PeerConnectionFactory.Options())
            .createPeerConnectionFactory()
        videoSource = factory.createVideoSource(false)
        videoTrack = factory.createVideoTrack("sinaptic5g-camera", videoSource)
    }

    fun start(token: String) {
        if (started || bffBaseUrl.isBlank()) return
        started = true
        scope.launch {
            try {
                onState("WEBRTC: SDP HAZIRLANIYOR")
                negotiate(token)
                framePublishingEnabled = true
                onState("WEBRTC: P2P BAĞLI")
            } catch (error: Exception) {
                framePublishingEnabled = false
                started = false
                Log.w(TAG, "P2P negotiation failed: ${error.javaClass.simpleName}")
                onState("WEBRTC: KULLANILAMIYOR — YEREL SENTINEL AKTİF")
            }
        }
    }

    /** Copies CameraX YUV planes before ImageProxy is closed by local inference. */
    fun publish(image: ImageProxy) {
        if (!framePublishingEnabled || image.format != android.graphics.ImageFormat.YUV_420_888) return
        try {
            val i420 = JavaI420Buffer.allocate(image.width, image.height)
            copyPlane(image.planes[0], image.width, image.height, i420.dataY, i420.strideY)
            copyPlane(image.planes[1], (image.width + 1) / 2, (image.height + 1) / 2, i420.dataU, i420.strideU)
            copyPlane(image.planes[2], (image.width + 1) / 2, (image.height + 1) / 2, i420.dataV, i420.strideV)
            val frame = VideoFrame(i420, image.imageInfo.rotationDegrees, image.imageInfo.timestamp)
            videoSource.capturerObserver.onFrameCaptured(frame)
            frame.release()
        } catch (error: Exception) {
            Log.w(TAG, "Camera frame could not be published", error)
        }
    }

    private fun copyPlane(
        plane: ImageProxy.PlaneProxy,
        width: Int,
        height: Int,
        destination: ByteBuffer,
        destinationStride: Int,
    ) {
        val source = plane.buffer.duplicate()
        for (row in 0 until height) {
            val sourceRow = row * plane.rowStride
            val destinationRow = row * destinationStride
            for (column in 0 until width) {
                destination.put(destinationRow + column, source.get(sourceRow + column * plane.pixelStride))
            }
        }
    }

    private suspend fun negotiate(token: String) {
        val iceGatheringComplete = CompletableDeferred<Unit>()
        val observer = object : PeerConnection.Observer {
            override fun onSignalingChange(state: PeerConnection.SignalingState?) = Unit
            override fun onIceConnectionChange(state: PeerConnection.IceConnectionState?) {
                if (state == PeerConnection.IceConnectionState.FAILED) {
                    framePublishingEnabled = false
                    onState("WEBRTC: ICE HATASI — YEREL SENTINEL AKTİF")
                }
            }
            override fun onConnectionChange(state: PeerConnection.PeerConnectionState?) {
                if (state in setOf(
                        PeerConnection.PeerConnectionState.FAILED,
                        PeerConnection.PeerConnectionState.CLOSED,
                        PeerConnection.PeerConnectionState.DISCONNECTED,
                    )
                ) {
                    framePublishingEnabled = false
                    onState("WEBRTC: BAĞLANTI KOPTU — YEREL SENTINEL AKTİF")
                }
            }
            override fun onIceConnectionReceivingChange(receiving: Boolean) = Unit
            override fun onIceGatheringChange(state: PeerConnection.IceGatheringState?) {
                if (state == PeerConnection.IceGatheringState.COMPLETE) {
                    iceGatheringComplete.complete(Unit)
                }
            }
            override fun onIceCandidate(candidate: IceCandidate?) = Unit
            override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>?) = Unit
            override fun onAddStream(stream: MediaStream?) = Unit
            override fun onRemoveStream(stream: MediaStream?) = Unit
            override fun onDataChannel(channel: DataChannel?) = Unit
            override fun onRenegotiationNeeded() = Unit
            override fun onAddTrack(receiver: RtpReceiver?, streams: Array<out MediaStream>?) = Unit
        }
        val configuration = PeerConnection.RTCConfiguration(fetchIceServers(token))
        val createdPeer = factory.createPeerConnection(configuration, observer)
            ?: error("PeerConnection could not be created")
        peer = createdPeer
        createdPeer.addTrack(videoTrack, listOf("sinaptic5g"))

        val offer = createdPeer.createOfferAwait(MediaConstraints())
        createdPeer.setLocalDescriptionAwait(offer)
        withTimeout(20_000L) { iceGatheringComplete.await() }
        val localSdp = createdPeer.localDescription?.description ?: error("local SDP missing")
        require(localSdp.toByteArray().size <= 64 * 1024) { "SDP exceeds 64 KiB" }
        val signalingId = postOffer(token, localSdp)
        val answer = pollAnswer(token, signalingId)
        createdPeer.setRemoteDescriptionAwait(SessionDescription(SessionDescription.Type.ANSWER, answer))
    }

    private fun fetchIceServers(token: String): List<PeerConnection.IceServer> {
        val result = mutableListOf<PeerConnection.IceServer>()
        val request = Request.Builder()
            .url("${bffBaseUrl.trimEnd('/')}/webrtc/config")
            .header("Authorization", "Bearer $token")
            .get()
            .build()
        val entries = http.newCall(request).execute().use { response ->
            if (!response.isSuccessful) error("ICE config rejected: ${response.code}")
            JSONObject(response.body!!.string()).getJSONArray("ice_servers")
        }
        for (index in 0 until entries.length()) {
            val entry = entries.getJSONObject(index)
            val rawUrls = entry.get("urls")
            val urls = when (rawUrls) {
                is JSONArray -> (0 until rawUrls.length()).map { rawUrls.getString(it) }
                else -> listOf(rawUrls.toString())
            }
            val builder = PeerConnection.IceServer.builder(urls)
            entry.optString("username").takeIf { it.isNotBlank() }?.let(builder::setUsername)
            entry.optString("credential").takeIf { it.isNotBlank() }?.let(builder::setPassword)
            result += builder.createIceServer()
        }
        return result
    }

    private fun postOffer(token: String, sdp: String): String {
        val body = JSONObject()
            .put("device_id", deviceId)
            .put("type", "offer")
            .put("sdp", sdp)
            .toString()
            .toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
            .url("${bffBaseUrl.trimEnd('/')}/webrtc/offer")
            .header("Authorization", "Bearer $token")
            .header("X-Device-Id", deviceId)
            .post(body)
            .build()
        http.newCall(request).execute().use { response ->
            if (!response.isSuccessful) error("offer rejected: ${response.code}")
            return JSONObject(response.body!!.string()).getString("signaling_id")
        }
    }

    private suspend fun pollAnswer(token: String, signalingId: String): String {
        val deadline = System.currentTimeMillis() + SDP_TIMEOUT_MS
        val encoded = URLEncoder.encode(signalingId, Charsets.UTF_8.name())
        while (System.currentTimeMillis() < deadline) {
            val request = Request.Builder()
                .url("${bffBaseUrl.trimEnd('/')}/webrtc/answer?signaling_id=$encoded")
                .header("Authorization", "Bearer $token")
                .get()
                .build()
            http.newCall(request).execute().use { response ->
                when (response.code) {
                    200 -> return JSONObject(response.body!!.string()).getString("sdp")
                    202 -> Unit
                    410 -> error("signaling expired")
                    else -> error("answer poll failed: ${response.code}")
                }
            }
            delay(500)
        }
        error("answer timeout")
    }

    fun close() {
        framePublishingEnabled = false
        scope.cancel()
        peer?.close()
        peer?.dispose()
        videoTrack.dispose()
        videoSource.dispose()
        factory.dispose()
        http.dispatcher.executorService.shutdown()
    }
}

private suspend fun PeerConnection.createOfferAwait(constraints: MediaConstraints): SessionDescription =
    suspendCoroutine { continuation ->
        createOffer(object : SdpObserver {
            override fun onCreateSuccess(description: SessionDescription) = continuation.resume(description)
            override fun onCreateFailure(message: String) = continuation.resumeWithException(IllegalStateException(message))
            override fun onSetSuccess() = Unit
            override fun onSetFailure(message: String) = Unit
        }, constraints)
    }

private suspend fun PeerConnection.setLocalDescriptionAwait(description: SessionDescription): Unit =
    suspendCoroutine { continuation ->
        setLocalDescription(object : SdpObserver {
            override fun onSetSuccess() = continuation.resume(Unit)
            override fun onSetFailure(message: String) = continuation.resumeWithException(IllegalStateException(message))
            override fun onCreateSuccess(description: SessionDescription) = Unit
            override fun onCreateFailure(message: String) = Unit
        }, description)
    }

private suspend fun PeerConnection.setRemoteDescriptionAwait(description: SessionDescription): Unit =
    suspendCoroutine { continuation ->
        setRemoteDescription(object : SdpObserver {
            override fun onSetSuccess() = continuation.resume(Unit)
            override fun onSetFailure(message: String) = continuation.resumeWithException(IllegalStateException(message))
            override fun onCreateSuccess(description: SessionDescription) = Unit
            override fun onCreateFailure(message: String) = Unit
        }, description)
    }
