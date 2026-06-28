package com.sinaptic.tripwire.model

/**
 * Tek kare analiz sonucu.
 * Edge (cihaz üstü) ve takım GPU sunucusu sonuçlarını birleştirir.
 */
data class AnalysisResult(
    val frameId:      Long         = 0L,
    val riskScore:    Float        = 0f,         // 0-100
    val riskLevel:    RiskLevel    = RiskLevel.DUSUK,
    val detections:   List<Detection> = emptyList(),
    val licensePlate: String?      = null,        // plate detector + CRNN/CTC
    val vehicleType:  String?      = null,
    val vehicleColor: String?      = null,
    val speedKmh:     Float?       = null,        // yalnız kalibre kamerada
    val behaviorClass: String?     = null,        // CNN-LSTM çıktısı
    val processingMs: Float        = 0f,          // İşlem süresi
    val isQodActive:  Boolean      = false,
    val targetPresent: Boolean     = false,
    val vehicleApproaching: Boolean = false,
    val recognizabilityGap: Float  = 0f,
    val modelUncertainty: Float    = 0f,
    val mediaDegradation: Float    = 0f,
    val networkDegradation: Float  = 0f,
    val timestamp:    Long         = System.currentTimeMillis(),
)

/**
 * Tek tespit kutusu.
 */
data class Detection(
    val classId:    Int,
    val className:  String,
    val confidence: Float,
    val bbox:       BoundingBox,
    val trackId:    Int = -1,
)

/**
 * Normalize edilmiş sınır kutusu [0,1] aralığında.
 */
data class BoundingBox(
    val x1: Float,  // Sol
    val y1: Float,  // Üst
    val x2: Float,  // Sağ
    val y2: Float,  // Alt
) {
    val width:  Float get() = x2 - x1
    val height: Float get() = y2 - y1
    val centerX: Float get() = (x1 + x2) / 2f
    val centerY: Float get() = (y1 + y2) / 2f
    val pixelWidth: Int get() = (width * 1280).toInt()
}

/**
 * Risk seviyeleri.
 */
enum class RiskLevel {
    DUSUK,   // 0-44
    ORTA,    // 45-69
    KRITIK,  // 70-100
}

/** Risk skoru → RiskLevel dönüşümü */
fun Float.toRiskLevel(): RiskLevel = when {
    this >= 70f -> RiskLevel.KRITIK
    this >= 45f -> RiskLevel.ORTA
    else        -> RiskLevel.DUSUK
}
