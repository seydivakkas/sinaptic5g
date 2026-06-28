# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("sinaptic5g.ftr.preprocessor")

class Preprocessor:
    """Manages low-light enhancement (Zero-DCE) and camera lens undistortion."""
    
    def __init__(self,
                 frame_size: Tuple[int, int],
                 camera_mode: str = "front",
                 zero_dce_threshold: int = 80):
        self.width, self.height = frame_size
        self.camera_mode = camera_mode
        self.zero_dce_threshold = zero_dce_threshold
        
        # Load camera calibration parameters
        from src.tracking_pipeline import CameraCalibrator
        self.K, self.D = CameraCalibrator.load(frame_size)
        self.H_bev = CameraCalibrator.compute_bev_homography(camera_mode, frame_size)
        self.needs_undistort = bool(np.any(np.abs(self.D) > 1e-12))
        
        logger.info("Preprocessor initialized. Undistortion active: %s, Camera: %s",
                    self.needs_undistort, self.camera_mode)

    def enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """Applies Zero-DCE low-light enhancement if the frame is under the brightness threshold."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        
        if mean_brightness < self.zero_dce_threshold:
            from src.models.low_light.zero_dce import enhance as zero_dce_enhance
            logger.debug("Low light detected (mean=%.2f), running Zero-DCE enhancement", mean_brightness)
            return zero_dce_enhance(frame)
        return frame

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Runs the complete preprocessing pipeline (Low light enhance -> Undistort)."""
        enhanced = self.enhance_low_light(frame)
        if self.needs_undistort:
            return cv2.undistort(enhanced, self.K, self.D)
        return enhanced
