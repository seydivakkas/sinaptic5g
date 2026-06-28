package com.sinaptic.tripwire.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.sinaptic.tripwire.model.Detection

/**
 * DetectionOverlayView — Bounding Box Çizim Katmanı
 *
 * CameraX PreviewView'in üzerine konumlandırılır.
 * Tespit edilen nesnelerin sınır kutularını ve etiketlerini çizer.
 */
class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs:   AttributeSet? = null,
    defStyle: Int          = 0,
) : View(context, attrs, defStyle) {

    private val boxPaints = mapOf(
        "license_plate" to Paint().apply { color = Color.parseColor("#00D4FF"); style = Paint.Style.STROKE; strokeWidth = 3f },
        "phone"         to Paint().apply { color = Color.parseColor("#FF1744"); style = Paint.Style.STROKE; strokeWidth = 4f },
        "cigarette"     to Paint().apply { color = Color.parseColor("#FF9800"); style = Paint.Style.STROKE; strokeWidth = 3f },
        "toy"           to Paint().apply { color = Color.parseColor("#FF9800"); style = Paint.Style.STROKE; strokeWidth = 3f },
        "togg"          to Paint().apply { color = Color.parseColor("#00FF88"); style = Paint.Style.STROKE; strokeWidth = 3f },
        "driver_face"   to Paint().apply { color = Color.parseColor("#AAAAAA"); style = Paint.Style.STROKE; strokeWidth = 2f },
    )

    private val defaultPaint = Paint().apply {
        color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 2f
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        isFakeBoldText = true
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }

    private val bgPaint = Paint().apply { style = Paint.Style.FILL }

    private var detections: List<Detection> = emptyList()

    fun setDetections(list: List<Detection>) {
        detections = list
        invalidate()  // Yeniden çiz
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (det in detections) {
            val box = det.bbox
            val left   = box.x1 * width
            val top    = box.y1 * height
            val right  = box.x2 * width
            val bottom = box.y2 * height

            // Sınır kutusu
            val paint = boxPaints[det.className] ?: defaultPaint
            canvas.drawRect(left, top, right, bottom, paint)

            // Etiket arka planı
            val label = "${det.className} ${(det.confidence * 100).toInt()}%"
            val textWidth  = textPaint.measureText(label)
            val textHeight = textPaint.textSize
            bgPaint.color = Color.parseColor("#CC000000")
            canvas.drawRect(left, top - textHeight - 8, left + textWidth + 8, top, bgPaint)

            // Etiket metni
            textPaint.color = paint.color
            canvas.drawText(label, left + 4, top - 6, textPaint)

            // Track ID (varsa)
            if (det.trackId >= 0) {
                val trackLabel = "#${det.trackId}"
                textPaint.color = Color.parseColor("#AAAAAA")
                textPaint.textSize = 22f
                canvas.drawText(trackLabel, right - 40, bottom - 4, textPaint)
                textPaint.textSize = 28f
            }
        }
    }
}
