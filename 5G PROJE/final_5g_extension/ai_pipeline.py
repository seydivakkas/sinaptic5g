"""Takım GPU'sundaki canlı çıkarım görev yönlendiricisi.

Bu modül QoD kararı vermez. Ağır modelin koşullu çalışması hesaplama kararıdır;
QoD kararı yalnız ölçülmüş fayda artefaktını kullanan qod_orchestrator'dadır.
"""

import cv2
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Merkezi sınıf kaydından import (src/class_registry.py)
from src.class_registry import (
    LIVE_NAME_TO_ID as CUSTOM_CLASSES,
    LIVE_DISTRACTOR_CLASSES as DISTRACTOR_CLASSES,
    LIVE_CLASS_COLORS as CLASS_COLORS,
)



@dataclass
class Detection:
    """Tek bir nesne tespiti."""
    class_name:  str
    confidence:  float
    bbox:        tuple[int, int, int, int]  # (x1, y1, x2, y2)
    track_id:    Optional[int]  = None
    distance_m:  Optional[float] = None

    @property
    def pixel_width(self) -> int:
        """Bounding box piksel genişliği."""
        return self.bbox[2] - self.bbox[0]

    @property
    def pixel_height(self) -> int:
        """Bounding box piksel yüksekliği."""
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        return {
            "class":      self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox":       list(self.bbox),
            "track_id":   self.track_id,
            "distance_m": self.distance_m,
        }


@dataclass
class FrameAnalysis:
    """Bir kareye ait tüm analiz sonuçları."""
    frame_id:           int
    timestamp:          float
    detections:         list[Detection] = field(default_factory=list)
    license_plate_text: Optional[str]   = None
    speed_kmh:          Optional[float] = None
    risk_score:         float           = 0.0
    risk_level:         str             = "DÜŞÜK"
    processing_time_ms: float           = 0.0
    # Temporal / Davranış analizi
    behavior_class:     Optional[str]   = None   # 5 sınıf
    behavior_conf:      float           = 0.0
    ear_value:          Optional[float] = None
    mar_value:          Optional[float] = None

    @property
    def has_vehicle(self) -> bool:
        return any(d.class_name in ("togg", "vehicle") for d in self.detections)

    @property
    def has_distractor(self) -> bool:
        return any(d.class_name in DISTRACTOR_CLASSES for d in self.detections)

    def to_dict(self) -> dict:
        return {
            "frame_id":           self.frame_id,
            "timestamp":          self.timestamp,
            "license_plate":      self.license_plate_text,
            "speed_kmh":          self.speed_kmh,
            "risk_score":         self.risk_score,
            "risk_level":         self.risk_level,
            "detections":         [d.to_dict() for d in self.detections],
            "processing_time_ms": round(self.processing_time_ms, 2),
            "behavior_class":     self.behavior_class,
            "behavior_conf":      round(self.behavior_conf, 3),
        }


class TripWirePipeline:
    """
    Ana YZ pipeline orchestrator.

    Mimari:
    - Hafif model (YOLOv8-nano) her kareyi işler
    - Görsel/model belirsizliği varsa ağır model görevlendirilebilir
    - ByteTrack ile nesne takibi sürekli tutulur
    - Non-Maximum Suppression ile çift tespit önlenir
    """

    def __init__(
        self,
        light_model_path: str     = "models/yolov8n_tripwire.pt",
        heavy_model_path: str     = "models/yolov8l_tripwire.pt",
        temporal_model_path: str  = "runs/temporal/best.pt",
        conf_threshold:   float   = 0.5,
        iou_threshold:    float   = 0.45,
        plate_pixel_threshold: int = 50,
        distractor_conf:  float   = 0.6,
        togg_conf:        float   = 0.7,
    ):
        self.conf_threshold        = conf_threshold
        self.iou_threshold         = iou_threshold
        self.plate_pixel_threshold = plate_pixel_threshold
        self.distractor_conf       = distractor_conf
        self.togg_conf             = togg_conf
        self.frame_counter         = 0

        # YZ Modelleri
        self._light_model    = None
        self._heavy_model    = None
        self._temporal_model = None
        self._feature_buffer: list = []   # Son 16 kare özellik vektörü
        self._load_models(light_model_path, heavy_model_path, temporal_model_path)

    def _load_models(self, light_path: str, heavy_path: str, temporal_path: str):
        """YZ modellerini yükler. Hata durumunda graceful fallback."""
        try:
            from ultralytics import YOLO
            logger.info(f"Hafif model yükleniyor: {light_path}")
            self._light_model = YOLO(light_path)
            logger.info("Hafif model hazır")
        except Exception as e:
            logger.warning(f"Hafif model yüklenemedi ({e}) — demo modu aktif")

        try:
            from ultralytics import YOLO
            logger.info(f"Ağır model yükleniyor: {heavy_path}")
            self._heavy_model = YOLO(heavy_path)
            logger.info("Ağır model hazır")
        except Exception as e:
            logger.warning(f"Ağır model yüklenemedi ({e})")

        # ── Temporal CNN-LSTM ──────────────────────────────────────────────
        try:
            import torch
            from src.models.temporal.cnn_lstm import TemporalCNNLSTM, FEATURE_DIM
            if Path(temporal_path).exists():
                model = TemporalCNNLSTM()
                ckpt  = torch.load(temporal_path, map_location="cpu")
                model.load_state_dict(ckpt["model"])
                model.eval()
                self._temporal_model = model
                logger.info(f"Temporal model hazır: {temporal_path} (val_acc={ckpt.get('val_acc', '?'):.4f})")
            else:
                logger.info(f"Temporal checkpoint bulunamadı: {temporal_path}")
        except Exception as e:
            logger.warning(f"Temporal model yüklenemedi ({e})") 

    def process_frame(self, frame: np.ndarray) -> FrameAnalysis:
        """
        Bir kareyi işler ve FrameAnalysis döner.

        İş akışı:
        1. Hafif model ile hızlı tarama
        2. Sunucu model iyileştirmesi gereksinimini kontrol et
        3. Gerekiyorsa ağır model ile detaylı analiz
        4. Tespitleri birleştir (NMS tabanlı)
        """
        start_time = time.perf_counter()
        self.frame_counter += 1

        analysis = FrameAnalysis(
            frame_id  = self.frame_counter,
            timestamp = time.time(),
        )

        # ─── Model yoksa açıkça boş sonuç ────────────────────────────────
        if self._light_model is None:
            analysis.processing_time_ms = 5.0
            logger.debug("Model yüklü değil; tespit üretilmedi")
            return analysis

        # ─── Hafif Model Taraması ─────────────────────────────────────────
        try:
            light_results = self._light_model.track(
                frame,
                conf    = self.conf_threshold,
                iou     = self.iou_threshold,
                persist = True,    # ByteTrack ile kalıcı takip
                verbose = False,
            )
            light_detections = self._parse_results(light_results)
        except Exception as e:
            logger.error(f"Hafif model tarama hatası: {e}")
            light_detections = []

        analysis.detections = light_detections

        # ─── Koşullu GPU görev yönlendirmesi (QoD kararından bağımsız) ───
        refinement_needed, reason = self._needs_server_refinement(
            frame, light_detections
        )

        # ─── Ağır Model (hesaplama yönlendirmesi; QoD'den bağımsız) ─────
        if refinement_needed and self._heavy_model is not None:
            logger.info("Koşullu GPU iyileştirmesi: %s", reason)
            try:
                heavy_results = self._heavy_model(
                    frame,
                    conf    = 0.4,    # Daha düşük eşik (daha hassas)
                    iou     = self.iou_threshold,
                    verbose = False,
                )
                heavy_detections = self._parse_results(heavy_results)
                analysis.detections = self._merge_detections(
                    light_detections, heavy_detections
                )
            except Exception as e:
                logger.error(f"Ağır model hatası: {e}")

        # ─── Temporal Davranış Sınıflandırması ───────────────────────────
        if self._temporal_model is not None:
            try:
                behavior_class, behavior_conf = self._run_temporal(
                    analysis
                )
                analysis.behavior_class = behavior_class
                analysis.behavior_conf  = behavior_conf
            except Exception as e:
                logger.debug(f"Temporal model hatası: {e}")

        # ─── İşlem Süresi ────────────────────────────────────────────────
        analysis.processing_time_ms = (
            (time.perf_counter() - start_time) * 1000
        )

        return analysis

    def _run_temporal(
        self,
        analysis: "FrameAnalysis",
        seq_len: int = 16,
    ) -> tuple[str, float]:
        """
        Temporal CNN-LSTM ile sürücü davranışı sınıflandırır.

        Özellik vektörü (7 boyut):
            [EAR, MAR, speed_norm, head_angle_norm,
             has_phone, has_cigarette, speed_excess]

        Returns:
            (class_name, confidence) tuple'ı.
        """
        import torch
        from src.models.temporal.cnn_lstm import BEHAVIOR_CLASSES, FEATURE_DIM

        # Anlık özellik vektörü oluştur
        has_phone     = any(d.class_name == "phone"     for d in analysis.detections)
        has_cigarette = any(d.class_name == "cigarette" for d in analysis.detections)
        speed_norm    = min((analysis.speed_kmh or 0.0) / 120.0, 1.0)
        speed_excess  = max(0.0, ((analysis.speed_kmh or 0.0) - 50.0) / 50.0)
        ear           = analysis.ear_value or 0.30
        mar           = analysis.mar_value or 0.15

        feat = [
            ear,
            mar,
            speed_norm,
            0.1,            # head_angle_norm (MediaPipe olmadığında sıfır)
            float(has_phone),
            float(has_cigarette),
            min(speed_excess, 1.0),
        ]

        # Buffer'a ekle
        self._feature_buffer.append(feat)
        if len(self._feature_buffer) > seq_len:
            self._feature_buffer = self._feature_buffer[-seq_len:]

        # Yeterli geçmiş yoksa sonuç döndürme
        if len(self._feature_buffer) < seq_len:
            return "normal_surus", 0.0

        # İnference
        import torch
        seq = torch.tensor([self._feature_buffer], dtype=torch.float32)
        with torch.no_grad():
            logits = self._temporal_model(seq)          # (1, num_classes)
            probs  = torch.softmax(logits, dim=-1)[0]
            cls_id = probs.argmax().item()
            conf   = probs[cls_id].item()

        return BEHAVIOR_CLASSES[cls_id], conf

    def _parse_results(self, results) -> list[Detection]:
        """YOLO çıktısını Detection listesine dönüştürür."""
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf       = float(box.conf[0])
                cls_id     = int(box.cls[0])
                class_name = result.names.get(cls_id, f"class_{cls_id}")
                track_id   = (
                    int(box.id[0])
                    if box.id is not None
                    else None
                )

                det = Detection(
                    class_name = class_name,
                    confidence = conf,
                    bbox       = (x1, y1, x2, y2),
                    track_id   = track_id,
                )
                detections.append(det)

        return detections

    def _needs_server_refinement(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> tuple[bool, str]:
        """
        Yalnız model görev yönlendirmesi. QoD kararını bu fonksiyon vermez.

        Kural 1: Plaka çok küçük → OCR için yüksek çözünürlük gerekli
        Kural 2: Dikkat dağıtıcı nesne yüksek güvenle tespit edildi
        Kural 3: TOGG aracı yakın mesafede tespit edildi
        """
        for det in detections:
            # ── Kural 1: Plaka piksel boyutu ──────────────────────────────
            if det.class_name == "license_plate":
                if det.pixel_width < self.plate_pixel_threshold:
                    return (
                        True,
                        f"Plaka çok küçük ({det.pixel_width}px "
                        f"< {self.plate_pixel_threshold}px)"
                    )

            # ── Kural 2: Dikkat dağıtıcı nesne ────────────────────────────
            if (
                det.class_name in DISTRACTOR_CLASSES
                and det.confidence > self.distractor_conf
            ):
                return (
                    True,
                    f"Dikkat dağıtıcı: {det.class_name} "
                    f"({det.confidence:.2f})"
                )

            # ── Kural 3: TOGG aracı yaklaşıyor ────────────────────────────
            if (
                det.class_name == "togg"
                and det.confidence > self.togg_conf
            ):
                return True, f"TOGG aracı tespit edildi ({det.confidence:.2f})"

        return False, ""

    def _merge_detections(
        self,
        light: list[Detection],
        heavy: list[Detection],
    ) -> list[Detection]:
        """
        Hafif ve ağır model tespitlerini birleştirir.
        Ağır model daha yüksek önceliğe sahiptir (daha yüksek doğruluk).
        Track ID ile eşleştirme yapılır.
        """
        merged: dict = {}

        # Hafif model tespitlerini ekle
        for det in light:
            key = det.track_id if det.track_id is not None else id(det)
            merged[key] = det

        # Ağır model tespitleri → daha yüksek güvenliyse üzerine yaz
        for det in heavy:
            key = det.track_id if det.track_id is not None else id(det)
            if key in merged:
                if det.confidence > merged[key].confidence:
                    merged[key] = det  # Ağır model daha güvenilir
            else:
                merged[key] = det  # Yeni tespit

        return list(merged.values())

    def draw_annotations(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
    ) -> np.ndarray:
        """
        Kare üzerine tespit kutularını ve bilgileri çizer.

        Gösterilen bilgiler:
        - Bounding box + sınıf adı + güven skoru + track ID
        - Plaka metni (sol üst)
        - Hız (km/s)
        - Risk skoru (renk kodlu)
        - QoD durumu (sol alt)
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # ─── Tespit Kutuları ─────────────────────────────────────────────
        for det in analysis.detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_name, (128, 128, 128))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Etiket
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id is not None:
                label += f" [#{det.track_id}]"

            # Etiket arka planı (okunabilirlik)
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )[0]
            cv2.rectangle(
                annotated,
                (x1, y1 - text_size[1] - 6),
                (x1 + text_size[0] + 4, y1),
                color, -1,
            )
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1,
            )

        # ─── Plaka Metni ─────────────────────────────────────────────────
        if analysis.license_plate_text:
            cv2.putText(
                annotated,
                f"Plaka: {analysis.license_plate_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2,
            )

        # ─── Hız ─────────────────────────────────────────────────────────
        if analysis.speed_kmh is not None:
            spd_color = (0, 255, 255) if analysis.speed_kmh <= 50 else (0, 0, 255)
            cv2.putText(
                annotated,
                f"Hiz: {analysis.speed_kmh:.1f} km/s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                spd_color, 2,
            )

        # ─── Risk Skoru ───────────────────────────────────────────────────
        score = analysis.risk_score
        if score < 30:
            risk_color = (0, 200, 0)    # Yeşil
        elif score < 70:
            risk_color = (0, 165, 255)  # Turuncu
        else:
            risk_color = (0, 0, 255)    # Kırmızı

        cv2.putText(
            annotated,
            f"Risk: {score:.0f}/100 ({analysis.risk_level})",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            risk_color, 2,
        )

        # ─── FPS ve İşlem Süresi ──────────────────────────────────────────
        cv2.putText(
            annotated,
            f"#{analysis.frame_id} | {analysis.processing_time_ms:.1f}ms",
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (150, 150, 150), 1,
        )

        return annotated


if __name__ == "__main__":
    # Modül yükleme testi
    print("TripWire AI Pipeline")
    print(f"Özel sınıflar: {list(CUSTOM_CLASSES.keys())}")
    print(f"Dikkat dağıtıcı sınıflar: {DISTRACTOR_CLASSES}")

    # Demo pipeline oluştur (model olmadan)
    pipeline = TripWirePipeline(
        light_model_path="models/yolov8n_tripwire.pt",
        heavy_model_path="models/yolov8l_tripwire.pt",
    )

    # Test karesi
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    analysis = pipeline.process_frame(dummy_frame)
    print(f"Test analizi: Frame #{analysis.frame_id}, {analysis.processing_time_ms:.1f}ms")
