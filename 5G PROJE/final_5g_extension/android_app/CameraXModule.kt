/*
 * ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
 * Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
 */

package com.sinaptic.tripwire.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.concurrent.ExecutorService

class CameraXModule(
    private val executor: ExecutorService,
    private val frameListener: (ImageProxy) -> Unit
) : ImageAnalysis.Analyzer {

    /**
     * Extracts live video frames from CameraX stream at 30 FPS and forwards them
     * to the TFLite inference runtime and the WebRTC MEC offloader.
     */
    override fun analyze(image: ImageProxy) {
        // Forward image frame to analyzer
        frameListener(image)
    }
}
