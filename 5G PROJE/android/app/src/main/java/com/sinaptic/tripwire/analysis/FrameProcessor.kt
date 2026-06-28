package com.sinaptic.tripwire.analysis

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import com.sinaptic.tripwire.model.AnalysisResult
import com.sinaptic.tripwire.model.BoundingBox
import com.sinaptic.tripwire.model.Detection
import com.sinaptic.tripwire.model.toRiskLevel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest

/**
 * Lightweight Android sentinel. LiteRT/TFLite + NNAPI is the single mobile
 * inference backend. Expensive plate, cabin and temporal tasks stay on the
 * team GPU node.
 */
class FrameProcessor(private val context: Context) {

    companion object {
        private const val TAG = "FrameProcessor"
        private const val MANIFEST_NAME = "model_manifest.json"
        private const val INPUT_SIZE = 640
        private const val CONFIDENCE_THRESHOLD = 0.45f
        private const val NMS_THRESHOLD = 0.45f

        val CLASS_NAMES = listOf(
            "license_plate", "phone", "cabin", "laptop", "teknocan", "vehicle"
        )
        private val CLASS_RISK = mapOf("phone" to 30f, "laptop" to 10f)
    }

    private var nnApiDelegate: NnApiDelegate? = null
    private var interpreter: Interpreter? = null
    private var manifest: ModelManifest? = null
    private var previousVehicleArea = 0f

    private data class ModelManifest(
        val modelAsset: String,
        val artifactSha256: String,
        val inputShape: IntArray,
        val outputShape: IntArray,
        val labels: List<String>,
        val domainMapping: Map<Int, String>,
    )

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val loadedManifest = loadManifest()
            require(loadedManifest.domainMapping.values.all { it in CLASS_NAMES }) {
                "Manifest contains an unknown domain class"
            }
            require(assetSha256(loadedManifest.modelAsset) == loadedManifest.artifactSha256) {
                "TFLite asset SHA-256 does not match manifest"
            }
            val options = Interpreter.Options().apply { setNumThreads(4) }
            try {
                nnApiDelegate = NnApiDelegate()
                options.addDelegate(nnApiDelegate)
            } catch (error: Exception) {
                Log.w(TAG, "NNAPI unavailable; TFLite CPU fallback: ${error.message}")
            }
            val loadedInterpreter = Interpreter(loadMappedModel(loadedManifest.modelAsset), options)
            require(loadedInterpreter.getInputTensor(0).shape().contentEquals(loadedManifest.inputShape)) {
                "TFLite input tensor does not match manifest"
            }
            require(loadedInterpreter.getOutputTensor(0).shape().contentEquals(loadedManifest.outputShape)) {
                "TFLite output tensor does not match manifest"
            }
            manifest = loadedManifest
            interpreter = loadedInterpreter
            Log.i(TAG, "LiteRT/TFLite vehicle sentinel loaded and hash verified")
        } catch (error: Exception) {
            Log.e(TAG, "TFLite model unavailable; sentinel will report no target", error)
            interpreter = null
        }
    }

    private fun loadManifest(): ModelManifest {
        val text = context.assets.open(MANIFEST_NAME).bufferedReader(Charsets.UTF_8).use { it.readText() }
        val json = JSONObject(text)
        val labelsJson = json.getJSONArray("labels")
        val labels = (0 until labelsJson.length()).map { labelsJson.getString(it) }
        val mappingJson = json.getJSONObject("domain_mapping")
        val mapping = mappingJson.keys().asSequence().associate { key ->
            key.toInt() to mappingJson.getString(key)
        }
        fun intArray(name: String): IntArray {
            val array = json.getJSONArray(name)
            return IntArray(array.length()) { array.getInt(it) }
        }
        return ModelManifest(
            modelAsset = json.getString("model_asset"),
            artifactSha256 = json.getString("artifact_sha256"),
            inputShape = intArray("input_shape"),
            outputShape = intArray("output_shape"),
            labels = labels,
            domainMapping = mapping,
        )
    }

    private fun assetSha256(assetName: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        context.assets.open(assetName).use { stream ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    private fun loadMappedModel(assetName: String): MappedByteBuffer {
        val descriptor = context.assets.openFd(assetName)
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength,
            )
        }
    }

    fun isModelReady(): Boolean = interpreter != null

    fun processFrame(image: ImageProxy): AnalysisResult = processFrame(image.toBitmap())

    fun processFrame(bitmap: Bitmap): AnalysisResult {
        val startedAt = System.nanoTime()
        val detections = interpreter?.let { runYolo(bitmap, it) } ?: emptyList()
        val vehicle = detections.maxByOrNull {
            if (it.className == "vehicle") it.bbox.width * it.bbox.height else -1f
        }
        val vehicleArea = vehicle?.let { it.bbox.width * it.bbox.height } ?: 0f
        val approaching = vehicle != null && previousVehicleArea > 0f && vehicleArea > previousVehicleArea * 1.05f
        if (vehicle != null) previousVehicleArea = vehicleArea

        val plateConfidence = detections
            .filter { it.className == "license_plate" }
            .maxOfOrNull { it.confidence } ?: 0f
        val maxConfidence = detections.maxOfOrNull { it.confidence } ?: 0f
        val brightness = calculateBrightness(bitmap)
        val mediaDegradation = ((90f - brightness) / 90f).coerceIn(0f, 1f)
        val riskScore = detections.sumOf { (CLASS_RISK[it.className] ?: 0f).toDouble() }
            .toFloat().coerceIn(0f, 100f)
        val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000f

        return AnalysisResult(
            riskScore = riskScore,
            riskLevel = riskScore.toRiskLevel(),
            detections = detections,
            behaviorClass = null,
            processingMs = elapsedMs,
            targetPresent = vehicle != null,
            vehicleApproaching = approaching,
            recognizabilityGap = if (vehicle == null) 0f else (1f - plateConfidence).coerceIn(0f, 1f),
            modelUncertainty = (1f - maxConfidence).coerceIn(0f, 1f),
            mediaDegradation = mediaDegradation,
            networkDegradation = 0f,
        )
    }

    private fun runYolo(bitmap: Bitmap, model: Interpreter): List<Detection> {
        return try {
            val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            val input = createInputBuffer(resized, model)
            val outputTensor = model.getOutputTensor(0)
            val output = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())
            model.run(input, output)
            parseOutput(output, outputTensor.shape(), outputTensor.dataType(), bitmap.width, bitmap.height)
        } catch (error: Exception) {
            Log.e(TAG, "TFLite inference failed", error)
            emptyList()
        }
    }

    private fun createInputBuffer(bitmap: Bitmap, model: Interpreter): ByteBuffer {
        val tensor = model.getInputTensor(0)
        val buffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (pixel in pixels) {
            val r = pixel shr 16 and 0xFF
            val g = pixel shr 8 and 0xFF
            val b = pixel and 0xFF
            when (tensor.dataType()) {
                DataType.FLOAT32 -> {
                    buffer.putFloat(r / 255f)
                    buffer.putFloat(g / 255f)
                    buffer.putFloat(b / 255f)
                }
                DataType.UINT8 -> {
                    buffer.put(r.toByte()); buffer.put(g.toByte()); buffer.put(b.toByte())
                }
                DataType.INT8 -> {
                    buffer.put((r - 128).toByte()); buffer.put((g - 128).toByte()); buffer.put((b - 128).toByte())
                }
                else -> error("Unsupported input tensor type: ${tensor.dataType()}")
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun parseOutput(
        output: ByteBuffer,
        shape: IntArray,
        dataType: DataType,
        imageWidth: Int,
        imageHeight: Int,
    ): List<Detection> {
        require(shape.size == 3 && shape[0] == 1) { "Expected YOLO rank-3 output" }
        val channelFirst = shape[1] < shape[2]
        val channels = if (channelFirst) shape[1] else shape[2]
        val boxes = if (channelFirst) shape[2] else shape[1]
        val values = FloatArray(channels * boxes)
        output.rewind()
        when (dataType) {
            DataType.FLOAT32 -> output.asFloatBuffer().get(values)
            DataType.UINT8 -> for (index in values.indices) values[index] = (output.get().toInt() and 0xFF) / 255f
            DataType.INT8 -> for (index in values.indices) values[index] = (output.get().toInt() + 128) / 255f
            else -> error("Unsupported output tensor type: $dataType")
        }
        fun value(channel: Int, box: Int): Float =
            if (channelFirst) values[channel * boxes + box] else values[box * channels + channel]

        val candidates = mutableListOf<Detection>()
        val loadedManifest = manifest ?: return emptyList()
        val classCount = minOf(channels - 4, loadedManifest.labels.size)
        for (box in 0 until boxes) {
            var bestRawClass = -1
            var bestConfidence = CONFIDENCE_THRESHOLD
            // Pick the model's winning raw class first. Only then apply the
            // manifest domain mapping, so the other 77 COCO classes are
            // dropped instead of being misreported as vehicles.
            for (classIndex in 0 until classCount) {
                val confidence = value(4 + classIndex, box)
                if (confidence > bestConfidence) {
                    bestConfidence = confidence
                    bestRawClass = classIndex
                }
            }
            if (bestRawClass < 0) continue
            val domainClass = loadedManifest.domainMapping[bestRawClass] ?: continue
            val cx = value(0, box) / INPUT_SIZE
            val cy = value(1, box) / INPUT_SIZE
            val width = value(2, box) / INPUT_SIZE
            val height = value(3, box) / INPUT_SIZE
            candidates += Detection(
                classId = CLASS_NAMES.indexOf(domainClass),
                className = domainClass,
                confidence = bestConfidence,
                bbox = BoundingBox(
                    x1 = (cx - width / 2).coerceIn(0f, 1f),
                    y1 = (cy - height / 2).coerceIn(0f, 1f),
                    x2 = (cx + width / 2).coerceIn(0f, 1f),
                    y2 = (cy + height / 2).coerceIn(0f, 1f),
                ),
            )
        }
        return nonMaximumSuppression(candidates)
    }

    private fun nonMaximumSuppression(input: List<Detection>): List<Detection> {
        val kept = mutableListOf<Detection>()
        for (candidate in input.sortedByDescending { it.confidence }) {
            if (kept.none { intersectionOverUnion(it.bbox, candidate.bbox) > NMS_THRESHOLD }) {
                kept += candidate
            }
        }
        return kept
    }

    private fun intersectionOverUnion(a: BoundingBox, b: BoundingBox): Float {
        val intersectionWidth = (minOf(a.x2, b.x2) - maxOf(a.x1, b.x1)).coerceAtLeast(0f)
        val intersectionHeight = (minOf(a.y2, b.y2) - maxOf(a.y1, b.y1)).coerceAtLeast(0f)
        val intersection = intersectionWidth * intersectionHeight
        val union = a.width * a.height + b.width * b.height - intersection
        return if (union > 0f) intersection / union else 0f
    }

    private fun calculateBrightness(bitmap: Bitmap): Float {
        var sum = 0.0
        var count = 0
        for (y in 0 until bitmap.height step 8) {
            for (x in 0 until bitmap.width step 8) {
                val pixel = bitmap.getPixel(x, y)
                sum += 0.299 * (pixel shr 16 and 0xFF) +
                    0.587 * (pixel shr 8 and 0xFF) +
                    0.114 * (pixel and 0xFF)
                count++
            }
        }
        return if (count == 0) 128f else (sum / count).toFloat()
    }

    fun release() {
        interpreter?.close()
        nnApiDelegate?.close()
    }
}
