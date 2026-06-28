package com.sinaptic.tripwire

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.media.MediaPlayer
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.sinaptic.tripwire.analysis.FrameProcessor
import com.sinaptic.tripwire.api.QoDManager
import com.sinaptic.tripwire.api.NumberVerificationManager
import com.sinaptic.tripwire.api.TripWireWebSocket
import com.sinaptic.tripwire.api.WebRtcMediaClient
import com.sinaptic.tripwire.databinding.ActivityMainBinding
import com.sinaptic.tripwire.model.AnalysisResult
import com.sinaptic.tripwire.model.RiskLevel
import com.sinaptic.tripwire.db.DetectionRepository
import kotlinx.coroutines.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.UUID

/**
 * MainActivity — Sinaptic5G TripWire Ana Ekranı
 *
 * Sorumluluklar:
 * - CameraX görüntüsünde LiteRT/TFLite + NNAPI çıkarımı yapar
 * - Ölçülmüş backend fayda kapısı için yaklaşma/görsel sinyalleri üretir
 * - WebSocket'ten yalnız kimlikli sonuç/telemetri alır
 * - Anlık tespitleri ve yerel uyarı skorunu UI'da gösterir
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "TripWire"
        private const val MAX_ANALYSIS_FPS = 15
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var frameProcessor: FrameProcessor
    private lateinit var qodManager: QoDManager
    private lateinit var numberVerification: NumberVerificationManager
    private lateinit var webSocket: TripWireWebSocket
    private lateinit var webRtcMedia: WebRtcMediaClient
    private lateinit var repository: DetectionRepository

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // QoD durumu
    private var qodActive = false
    private var lastQodTrigger = 0L

    // İstatistikler
    private var frameCount = 0
    private var detectionCount = 0
    private var lastAnalysisAtNanos = 0L
    private val deviceId: String by lazy {
        Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID)
            ?.takeIf { it.isNotBlank() }
            ?: UUID.randomUUID().toString()
    }

    // Video Oynatma & Analiz
    private var mediaPlayer: MediaPlayer? = null
    private var videoJob: Job? = null
    private var selectedSource = 0 // 0: Kamera, 1: Video 1, 2: Video 2, 3: Video 3
    private val videoFiles = listOf("video_1.mp4", "video_2.mp4", "video_3.mp4")

    // ─── İzin İsteme ──────────────────────────────────────────────────────
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Kamera izni gerekli", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    private val authorizationLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode != RESULT_OK || result.data == null) {
            binding.txtAuthStatus.text = "DOĞRULAMA BAŞARISIZ"
            return@registerForActivityResult
        }
        numberVerification.handleAuthorizationResult(result.data!!) { tokenReady ->
            scope.launch {
                if (!tokenReady) {
                    binding.txtAuthStatus.text = "KISA ÖMÜRLÜ TOKEN ALINAMADI"
                    return@launch
                }
                val verified = numberVerification.verify(
                    binding.editPhone.text.toString().trim(),
                    deviceId,
                )
                binding.txtAuthStatus.text = if (verified) "NUMARA DOĞRULANDI" else "NUMARA DOĞRULANAMADI"
                if (verified) {
                    numberVerification.appSessionToken()?.let(::connectWebSocket)
                }
            }
        }
    }

    // ─── Yaşam Döngüsü ────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Bileşenleri başlat
        frameProcessor = FrameProcessor(this)
        qodManager     = QoDManager(this)
        numberVerification = NumberVerificationManager(this)
        webSocket      = TripWireWebSocket(
            serverUrl  = BuildConfig.MEDIA_BASE_URL
                .replaceFirst("https://", "wss://")
                .replaceFirst("http://", "ws://") + "/ws/telemetry",
            deviceId   = deviceId,
            onResult   = ::onAnalysisResult,
            onConnectionState = ::onTelemetryConnectionState,
        )
        webRtcMedia = WebRtcMediaClient(
            context = this,
            bffBaseUrl = BuildConfig.API_BASE_URL,
            deviceId = deviceId,
            onState = { state -> runOnUiThread { binding.txtConnection.text = state } },
        )
        repository = DetectionRepository(this)

        // İzin kontrol
        if (hasCameraPermission()) {
            startCamera()
        } else {
            permissionLauncher.launch(
                arrayOf(
                    Manifest.permission.CAMERA,
                )
            )
        }

        setupUI()
        setupVideoTextureListener()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopVideoPlayback()
        scope.cancel()
        cameraExecutor.shutdown()
        webSocket.disconnect()
        webRtcMedia.close()
        qodManager.cleanup()
        numberVerification.close()
        frameProcessor.release()
    }

    // ─── Kamera ───────────────────────────────────────────────────────────

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Önizleme
            val preview = Preview.Builder()
                .setTargetResolution(android.util.Size(1280, 720))
                .build()
                .also { it.setSurfaceProvider(binding.viewFinder.surfaceProvider) }

            // Frame analizi
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ::analyzeFrame)
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
                Log.d(TAG, "Kamera baslatildi: 720p")
            } catch (e: Exception) {
                Log.e(TAG, "Kamera baslatma hatasi: $e")
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {
        if (selectedSource != 0) {
            image.close()
            return
        }
        frameCount++
        webRtcMedia.publish(image)

        val now = System.nanoTime()
        val minimumInterval = 1_000_000_000L / MAX_ANALYSIS_FPS
        if (now - lastAnalysisAtNanos < minimumInterval) {
            image.close()
            return
        }
        lastAnalysisAtNanos = now

        scope.launch(Dispatchers.IO) {
            try {
                // Cihaz üzerinde YOLOv8-nano TFLite çalıştır
                val localResult = frameProcessor.processFrame(image)

                // Risk skoru hesapla
                val riskScore = localResult.riskScore

                // QoD yalnız sert yaklaşma kapısı ve ölçülmüş backend fayda modeliyle açılır.
                if (shouldRequestQod(localResult) && !qodActive) {
                    triggerQoD(localResult)
                }

                // UI güncelle
                withContext(Dispatchers.Main) {
                    updateUI(localResult)
                }

                // Yerel veritabanına kaydet (kritik tespitler)
                if (riskScore >= 70f) {
                    repository.saveDetection(localResult)
                }

            } catch (e: Exception) {
                Log.e(TAG, "Frame analiz hatasi: $e")
            } finally {
                image.close()
            }
        }
    }

    // ─── QoD Yönetimi ────────────────────────────────────────────────────

    private fun triggerQoD(result: AnalysisResult) {
        val now = System.currentTimeMillis()
        // Son QoD tetiklemeden 30 saniye geçmeden tekrar tetikleme
        if (now - lastQodTrigger < 30_000) return

        lastQodTrigger = now
        scope.launch {
            try {
                val token = numberVerification.appSessionToken()
                val phone = numberVerification.verifiedPhone()
                if (token == null || phone == null) {
                    binding.qodIndicator.text = "5G BEST EFFORT — NUMARA DOĞRULANMADI"
                    return@launch
                }
                val sessionId = qodManager.startSession(
                    phoneNumber = phone,
                    signals = result,
                    accessToken = token,
                    deviceId = deviceId,
                )
                if (sessionId != null) {
                    qodActive = true
                    Log.i(TAG, "QoD aktif: $sessionId")
                    withContext(Dispatchers.Main) {
                        binding.qodIndicator.text = "5G QoD AKTİF"
                        binding.qodIndicator.setBackgroundResource(R.color.color_qod_active)
                    }
                    // 300 saniye sonra QoD'u kapat
                    delay(300_000L)
                    qodManager.stopSession(sessionId, deviceId, token)
                    qodActive = false
                    binding.qodIndicator.text = "5G BEST EFFORT"
                    binding.qodIndicator.setBackgroundResource(R.color.panel_section)
                } else {
                    binding.qodIndicator.text = "5G BEST EFFORT — QoD AÇILMADI"
                    binding.qodIndicator.setBackgroundResource(R.color.panel_section)
                }
            } catch (e: Exception) {
                Log.e(TAG, "QoD hatasi: $e")
                qodActive = false
                binding.qodIndicator.text = "5G BEST EFFORT — QoD KULLANILAMIYOR"
                binding.qodIndicator.setBackgroundResource(R.color.panel_section)
            }
        }
    }

    private fun shouldRequestQod(result: AnalysisResult): Boolean =
        result.targetPresent && result.vehicleApproaching

    // ─── WebSocket Sonuçları ──────────────────────────────────────────────

    private fun onAnalysisResult(eventId: String, result: AnalysisResult) {
        detectionCount += result.detections.size
        if (result.riskScore >= 70f) {
            scope.launch(Dispatchers.IO) { repository.saveDetection(result, eventId) }
        }
        scope.launch(Dispatchers.Main) {
            // Backend sonuçlarını UI'ya yansıt
            binding.txtPlate.text = result.licensePlate ?: "—"
            binding.txtVehicle.text = listOfNotNull(result.vehicleType, result.vehicleColor)
                .takeIf { it.isNotEmpty() }
                ?.joinToString(" / ") ?: "—"
            binding.txtEventTime.text = java.text.SimpleDateFormat(
                "HH:mm:ss.SSS",
                java.util.Locale.getDefault(),
            ).format(java.util.Date(result.timestamp))
            binding.txtSpeed.text = result.speedKmh?.let { "${it.toInt()} km/s" } ?: "—"

            val riskColor = when (result.riskLevel) {
                RiskLevel.DUSUK   -> R.color.color_risk_low
                RiskLevel.ORTA    -> R.color.color_risk_medium
                RiskLevel.KRITIK  -> R.color.color_risk_critical
            }
            binding.riskBar.progress = result.riskScore.toInt()
            binding.riskBar.progressTintList =
                ContextCompat.getColorStateList(this@MainActivity, riskColor)
            binding.txtRiskScore.text = "${result.riskScore.toInt()}/100"
        }
    }

    private fun onTelemetryConnectionState(state: TripWireWebSocket.ConnectionState) {
        runOnUiThread {
            binding.txtConnection.text = when (state) {
                TripWireWebSocket.ConnectionState.CONNECTED -> "TELEMETRİ: BAĞLI"
                TripWireWebSocket.ConnectionState.CONNECTING -> "TELEMETRİ: BAĞLANIYOR"
                TripWireWebSocket.ConnectionState.RECONNECTING -> "TELEMETRİ: YENİDEN BAĞLANIYOR"
                TripWireWebSocket.ConnectionState.FAILED -> "TELEMETRİ: BAĞLANTI HATASI"
                TripWireWebSocket.ConnectionState.DISCONNECTED -> "TELEMETRİ: BAĞLI DEĞİL"
            }
        }
    }

    // ─── UI Kurulumu ──────────────────────────────────────────────────────

    private fun setupUI() {
        binding.txtInference.text = if (frameProcessor.isModelReady()) {
            "ÇIKARIM: LITERT/TFLITE HAZIR"
        } else {
            "ÇIKARIM: MODEL EKSİK — SONUÇ ÜRETİLMİYOR"
        }
        binding.btnVerify.setOnClickListener {
            try {
                authorizationLauncher.launch(numberVerification.authorizationIntent())
            } catch (error: Exception) {
                binding.txtAuthStatus.text = "OIDC YAPILANDIRMASI EKSİK"
                Log.w(TAG, "Number Verification başlatılamadı", error)
            }
        }
        binding.btnStop.setOnClickListener {
            scope.cancel()
            finish()
        }

        // Giriş Kaynağı Spinner Kurulumu
        val sources = listOf(
            "Canlı Kamera",
            "Test Videosu 1 (Plaka)",
            "Test Videosu 2 (Gece)",
            "Test Videosu 3 (Sürücü)"
        )
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, sources)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerSource.adapter = adapter

        binding.spinnerSource.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (selectedSource != position) {
                    selectedSource = position
                    handleSourceChange()
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // Her saniye istatistik güncelle
        scope.launch {
            while (isActive) {
                delay(1000)
                withContext(Dispatchers.Main) {
                    binding.txtStats.text =
                        "Kare: $frameCount | Tespit: $detectionCount | " +
                        "QoD: ${if (qodActive) "AKTIF" else "Kapalı"}"
                }
            }
        }
    }

    private fun updateUI(result: AnalysisResult) {
        // Cihaz üzerindeki anlık sonuçları göster
        val riskText = "${result.riskScore.toInt()}/100"
        binding.txtLocalRisk.text = riskText
        binding.txtBehavior.text = result.behaviorClass ?: "—"
        binding.detectionOverlay.setDetections(result.detections)
    }

    private fun connectWebSocket(accessToken: String) {
        if (BuildConfig.MEDIA_BASE_URL.isBlank()) {
            binding.txtConnection.text = "TELEMETRİ: MEDIA_BASE_URL EKSİK"
            return
        }
        scope.launch {
            webSocket.connect(accessToken)
            webRtcMedia.start(accessToken)
        }
    }

    // ─── Yardımcı ─────────────────────────────────────────────────────────

    // ─── Video Yönetim ve Analiz Metotları ─────────────────────────────────

    private fun handleSourceChange() {
        if (selectedSource == 0) {
            // Kamera moduna geç
            binding.videoView.visibility = View.GONE
            binding.viewFinder.visibility = View.VISIBLE
            stopVideoPlayback()
            startCamera()
        } else {
            // Video moduna geç
            binding.viewFinder.visibility = View.GONE
            binding.videoView.visibility = View.VISIBLE
            
            try {
                val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()
            } catch (e: Exception) {
                Log.w(TAG, "Kamera durdurma hatasi: $e")
            }

            val fileName = videoFiles[selectedSource - 1]
            playVideoFromAssets(fileName)
        }
    }

    private fun playVideoFromAssets(fileName: String) {
        stopVideoPlayback()
        mediaPlayer = MediaPlayer().apply {
            try {
                val afd = assets.openFd(fileName)
                setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                afd.close()
                isLooping = true
                
                if (binding.videoView.isAvailable) {
                    setSurface(Surface(binding.videoView.surfaceTexture))
                }
                
                setOnPreparedListener {
                    it.start()
                    startVideoFrameAnalysis()
                }
                prepareAsync()
            } catch (e: Exception) {
                Log.e(TAG, "Video oynatma hatasi: $e")
            }
        }
    }

    private fun stopVideoPlayback() {
        videoJob?.cancel()
        videoJob = null
        try {
            mediaPlayer?.stop()
            mediaPlayer?.release()
        } catch (e: Exception) {
            Log.w(TAG, "Player release hatasi: $e")
        }
        mediaPlayer = null
    }

    private fun startVideoFrameAnalysis() {
        videoJob?.cancel()
        videoJob = scope.launch(Dispatchers.IO) {
            val delayMs = 1000L / MAX_ANALYSIS_FPS
            while (isActive) {
                val t0 = System.currentTimeMillis()
                
                if (selectedSource > 0) {
                    var bitmap: android.graphics.Bitmap? = null
                    withContext(Dispatchers.Main) {
                        if (binding.videoView.isAvailable) {
                            bitmap = binding.videoView.bitmap
                        }
                    }
                    
                    val bmp = bitmap
                    if (bmp != null) {
                        try {
                            // Cihaz üzerinde yalnız LiteRT/TFLite sentinel çalıştır
                            val localResult = frameProcessor.processFrame(bmp)
                            val riskScore = localResult.riskScore

                            if (shouldRequestQod(localResult) && !qodActive) {
                                triggerQoD(localResult)
                            }

                            // UI güncelle
                            withContext(Dispatchers.Main) {
                                updateUI(localResult)
                            }

                            // Yerel veritabanına kaydet (kritik tespitler)
                            if (riskScore >= 70f) {
                                repository.saveDetection(localResult)
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Video frame analiz hatasi: $e")
                        }
                    }
                }
                
                val elapsed = System.currentTimeMillis() - t0
                val sleep = (delayMs - elapsed).coerceAtLeast(0L)
                delay(sleep)
            }
        }
    }

    private fun setupVideoTextureListener() {
        binding.videoView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) {
                mediaPlayer?.setSurface(Surface(st))
            }
            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
                mediaPlayer?.setSurface(null)
                return true
            }
            override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
        }
    }

    // ─── Yardımcı ─────────────────────────────────────────────────────────

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

}
