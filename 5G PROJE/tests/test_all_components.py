# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
#
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
#
# Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
# amaçlı olarak paylaşılmıştır.
#
# YASAKLAR:
#   1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
#   2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
#   3. Alt lisanslanamaz, satılamaz veya devredilemez.
#   4. Tersine mühendislik yapılamaz.
#
# İZİN VERİLEN KULLANIM:
#   - GitHub üzerinde görüntüleme ve okuma.
#   - Kişisel öğrenim amacıyla kodu inceleme (kopyalamadan).
#
# YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
# İzin talepleri için: GitHub @seydivakkas

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from plate_ocr import PlateRecognizer
from src.models.low_light.zero_dce import enhance as zero_dce_enhance
from src.models.temporal.cnn_lstm import ONNXTemporalClassifier, FeatureExtractor
from src.tracking_pipeline import SinapticTracker, CameraCalibrator


def test_zero_dce_enhance_fallback():
    """Verifies Zero-DCE enhance function gracefully falls back or bypasses."""
    # Test bright frame (mean > 80) -> should bypass immediately
    bright_frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    enhanced = zero_dce_enhance(bright_frame)
    assert np.array_equal(enhanced, bright_frame)

    # Test dark frame (mean < 80) -> if model is absent, it should return original frame
    dark_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    enhanced_dark = zero_dce_enhance(dark_frame, model_path="non_existent_model.tflite")
    assert np.array_equal(enhanced_dark, dark_frame)


def test_plate_ocr_fallback():
    """Verifies PlateRecognizer handles missing models gracefully by returning fallback sentinel."""
    recognizer = PlateRecognizer(lprnet_model_path="absent.onnx", crnn_model_path="absent.onnx")
    dummy_plate = np.zeros((50, 150, 3), dtype=np.uint8)
    res = recognizer.recognize(dummy_plate)
    # When models are absent, it should fallback to voting/sentinel if present or None
    # In a fresh buffer, it returns None/voted which is None
    assert res is None


def test_cnn_lstm_classifier_and_extractor():
    """Verifies FeatureExtractor and ONNXTemporalClassifier fallback behavior."""
    extractor = FeatureExtractor()
    feat = extractor.extract(
        ear=0.28,
        mar=0.15,
        speed_px=5.0,
        angle_deg=10.0,
        has_phone=True,
        has_cigarette=False,
        speed_kmh=45.0,
        speed_limit_kmh=50.0
    )
    assert len(feat) == 7
    # Should work on numpy array representation
    assert float(feat[4]) == 1.0
    assert float(feat[5]) == 0.0

    classifier = ONNXTemporalClassifier(model_path="absent.onnx")
    # Missing session should cause predict to return None
    dummy_seq = np.zeros((16, 7), dtype=np.float32)
    res = classifier.predict(dummy_seq)
    assert res is None


def test_botsort_tracker_update():
    """Verifies BoTSORT tracker update loop with simple dummy detections."""
    H = np.eye(3, dtype=np.float32)
    tracker = SinapticTracker(camera_mode="front", bev_H=H, conf_thresh=0.45, track_buffer=10)

    # Create dummy frame and raw detection
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    class DummyDetection:
        def __init__(self, bbox, conf):
            self.bbox = bbox
            self.confidence = conf
            self.vehicle_type = "sedan"

    # Frame 1: Initial high conf detection
    det1 = DummyDetection((100, 100, 200, 200), 0.85)
    results = tracker.update(dummy_frame, [det1], 0.0)
    assert len(results) == 1
    assert results[0][1] == 1  # assigned track ID = 1

    # Frame 2: Sequence update with optical flow CMC and matching
    det2 = DummyDetection((102, 102, 202, 202), 0.88)
    results2 = tracker.update(dummy_frame, [det2], 0.04)
    assert len(results2) == 1
    assert results2[0][1] == 1  # should match track ID = 1
