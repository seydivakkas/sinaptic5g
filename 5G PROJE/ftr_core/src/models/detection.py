"""
detection.py — Sinaptic5G YOLO Nesne Tespiti Wrapper
=====================================================
Ultralytics YOLOv8/v10 modelini yükleyen, Türk trafik sınıflarına
göre head'ini ayarlayan wrapper sınıf.

Kullanım:
    from src.models.detection import YOLODetector

    detector = YOLODetector(model_path="models/yolov8n_tripwire.pt")
    detections = detector.detect(frame)
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Türk trafik sınıfları
TURKISH_TRAFFIC_CLASSES: list[str] = [
    "otomobil",
    "kamyon",
    "motosiklet",
    "otobus",
    "yaya",
    "trafik_isareti",
    "trafik_isigi",
    "TOGG",
    "license_plate",
    "phone",
    "cigarette",
    "driver_face",
]


@dataclass
class DetectionResult:
    """Tek bir tespit sonucu."""

    class_name: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    track_id: Optional[int] = None

    @property
    def pixel_width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def pixel_height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2,
        )


class YOLODetector:
    """Ultralytics YOLOv8 nesne tespiti wrapper.

    Türk trafik koşullarına özel sınıflar (otomobil, kamyon,
    motosiklet, otobüs, yaya, trafik_isareti, trafik_isigi, TOGG)
    ile fine-tune edilmiş modeli yükler ve çalıştırır.

    Kullanım:
        detector = YOLODetector("models/yolov8n_tripwire.pt", device="cpu")
        results = detector.detect(frame)
        results_timed = detector.detect_with_timing(frame)
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n_tripwire.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        target_latency_ms: float = 15.0,
        class_names: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            model_path: YOLO model dosya yolu (.pt).
            conf_threshold: Güven eşiği.
            iou_threshold: IoU NMS eşiği.
            device: Çalıştırma cihazı ("cpu", "0", "cuda").
            target_latency_ms: Hedef gecikme (loglama için).
            class_names: Sınıf isimleri (None → Türk trafik sınıfları).
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_latency_ms = target_latency_ms
        self.class_names = class_names or TURKISH_TRAFFIC_CLASSES

        self._model = None
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """YOLO modelini yükler."""
        try:
            from ultralytics import YOLO

            if not Path(model_path).exists():
                logger.warning(
                    f"Model dosyası bulunamadı: {model_path} — demo modu aktif"
                )
                return

            logger.info(f"YOLO modeli yükleniyor: {model_path}")
            self._model = YOLO(model_path)
            logger.info(f"✅ YOLO modeli hazır ({self.device})")
        except ImportError:
            logger.warning("ultralytics yüklü değil: pip install ultralytics")
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")

    @property
    def is_loaded(self) -> bool:
        """Model yüklü mü?"""
        return self._model is not None

    def detect(
        self,
        frame: np.ndarray,
        track: bool = False,
    ) -> list[DetectionResult]:
        """Tek frame üzerinde nesne tespiti yapar.

        Args:
            frame: BGR numpy array.
            track: ByteTrack ile takip uygula.

        Returns:
            DetectionResult listesi.
        """
        if not self.is_loaded:
            return []

        try:
            if track:
                results = self._model.track(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    persist=True,
                    verbose=False,
                )
            else:
                results = self._model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )
            return self._parse_results(results)
        except Exception as e:
            logger.error(f"Tespit hatası: {e}")
            return []

    def detect_with_timing(
        self,
        frame: np.ndarray,
        track: bool = False,
    ) -> tuple[list[DetectionResult], float]:
        """Tespit yapar ve gecikmeyi ölçer.

        Performans hedeflerini loglar:
        - YOLOv8-nano: ≤15ms
        - YOLOv8-large: ≤30ms

        Args:
            frame: BGR numpy array.
            track: ByteTrack takip.

        Returns:
            (detections, latency_ms) tuple'ı.
        """
        start = time.perf_counter()
        detections = self.detect(frame, track=track)
        latency_ms = (time.perf_counter() - start) * 1000

        # Performans hedefi loglama
        if latency_ms > self.target_latency_ms:
            logger.warning(
                f"⚠️ Gecikme hedefi aşıldı: {latency_ms:.1f}ms "
                f"(hedef: ≤{self.target_latency_ms}ms)"
            )
        else:
            logger.debug(
                f"Tespit: {len(detections)} nesne, "
                f"{latency_ms:.1f}ms (hedef: ≤{self.target_latency_ms}ms)"
            )

        return detections, latency_ms

    def _parse_results(self, results) -> list[DetectionResult]:
        """Ultralytics sonuçlarını DetectionResult'a dönüştürür."""
        detections: list[DetectionResult] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = result.names.get(cls_id, f"class_{cls_id}")
                track_id = int(box.id[0]) if box.id is not None else None

                detections.append(
                    DetectionResult(
                        class_name=class_name,
                        class_id=cls_id,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        track_id=track_id,
                    )
                )

        return detections


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = YOLODetector(device="cpu")
    print(f"Model yüklü: {detector.is_loaded}")
    print(f"Sınıflar: {detector.class_names}")
