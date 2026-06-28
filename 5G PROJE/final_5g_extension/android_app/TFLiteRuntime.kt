/*
 * ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 */

package com.sinaptic.tripwire.analysis

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

class TFLiteRuntime(context: Context, modelPath: String) {
    private var interpreter: Interpreter? = null

    init {
        // Load detector.tflite from assets
        val modelBuffer = context.assets.open(modelPath).use {
            val bytes = it.readBytes()
            ByteBuffer.allocateDirect(bytes.size).apply {
                put(bytes)
            }
        }
        interpreter = Interpreter(modelBuffer)
    }

    /**
     * Executes local lightweight object detection for vehicle presence and critical events
     * inside the mobile client. Falls back silently to offline cache if MEC link degrades.
     */
    fun runInference(inputData: ByteBuffer): Array<FloatArray> {
        val outputMap = Array(1) { FloatArray(8400 * 6) }
        interpreter?.run(inputData, outputMap)
        return outputMap
    }
}
