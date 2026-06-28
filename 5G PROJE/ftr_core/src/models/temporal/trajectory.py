"""
trajectory.py — Sinaptic5G Temporal Yörünge Takibi ve Anomali Tespiti
======================================================================
ByteTrack ile takip edilen nesnelerin zaman serisi yörüngelerini tutar,
hız anomalisi ve davranış değişikliği tespiti yapar.

Modül içeriği:
    - TrackState      : Tek nesnenin temporal durumu
    - TrajectoryBuffer: Çoklu nesne yörünge hafızası
    - TemporalAnomalyDetector: Hız/yön anomali tespiti
    - BehaviorClassifier     : EAR/MAR zaman serisinden uyuklama tespiti

Kullanım:
    from src.models.temporal.trajectory import TrajectoryBuffer, TemporalAnomalyDetector

    buffer = TrajectoryBuffer(max_track_age=60)
    anomaly_det = TemporalAnomalyDetector()

    # Her kare sonrası güncelle
    buffer.update(track_id=5, bbox=(x1, y1, x2, y2), frame_idx=n)

    # Anomali kontrolü
    result = anomaly_det.check(buffer.get(5))
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Veri Yapıları ────────────────────────────────────────────────────────────


@dataclass
class BBoxPoint:
    """Tek bir frame'deki bounding box gözlemi."""

    frame_idx: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        """Merkez x koordinatı."""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """Merkez y koordinatı."""
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class TrackState:
    """Tek bir tracked nesnenin temporal durumu.

    Her track_id için bir TrackState örneği tutulur.
    Yörünge, hız ve anomali bilgilerini içerir.
    """

    track_id: int
    class_name: str
    max_history: int = 64

    # Yörünge geçmişi (deque ile sınırlı boyut)
    history: deque = field(default_factory=lambda: deque(maxlen=64))

    # Türetilmiş metrikler
    last_speed_px_per_frame: float = 0.0
    speed_history: deque = field(default_factory=lambda: deque(maxlen=32))

    # Anomali bayrakları
    is_sudden_stop: bool = False
    is_sudden_acceleration: bool = False
    is_erratic_motion: bool = False

    # Frame sayacı (son güncelleme)
    last_seen_frame: int = -1

    def __post_init__(self):
        self.history = deque(maxlen=self.max_history)
        self.speed_history = deque(maxlen=32)

    def add_observation(self, bbox: BBoxPoint) -> None:
        """Yeni gözlem ekler ve hız hesaplar."""
        if self.history:
            prev = self.history[-1]
            dt = max(1, bbox.frame_idx - prev.frame_idx)
            dx = bbox.cx - prev.cx
            dy = bbox.cy - prev.cy
            speed = math.sqrt(dx**2 + dy**2) / dt
            self.last_speed_px_per_frame = speed
            self.speed_history.append(speed)
        self.history.append(bbox)
        self.last_seen_frame = bbox.frame_idx

    @property
    def age(self) -> int:
        """Kaç kare boyunca takip edildiği."""
        return len(self.history)

    @property
    def mean_speed(self) -> float:
        """Ortalama hız (px/frame)."""
        if not self.speed_history:
            return 0.0
        return float(np.mean(self.speed_history))

    @property
    def std_speed(self) -> float:
        """Hız standart sapması."""
        if len(self.speed_history) < 2:
            return 0.0
        return float(np.std(self.speed_history))

    @property
    def trajectory_points(self) -> list[tuple[float, float]]:
        """Yörünge merkez noktaları listesi."""
        return [(p.cx, p.cy) for p in self.history]

    @property
    def latest_bbox(self) -> Optional[BBoxPoint]:
        """Son gözlem."""
        return self.history[-1] if self.history else None

    def is_stale(self, current_frame: int, max_age: int = 30) -> bool:
        """Nesne yeterince uzun süre görülmedi mi?"""
        return (current_frame - self.last_seen_frame) > max_age


# ─── Yörünge Buffer ───────────────────────────────────────────────────────────


class TrajectoryBuffer:
    """Tüm aktif track'lerin yörüngelerini yönetir.

    Her track_id için TrackState örneği tutar.
    Eski (stale) track'leri otomatik temizler.

    Kullanım:
        buffer = TrajectoryBuffer(max_track_age=60)
        buffer.update(track_id=1, class_name="otomobil",
                      bbox=(10, 20, 100, 80), frame_idx=5, confidence=0.95)
        state = buffer.get(1)
    """

    def __init__(
        self,
        max_track_age: int = 60,
        max_history_per_track: int = 64,
    ) -> None:
        """
        Args:
            max_track_age: Kaç frame görünmezse track silinir.
            max_history_per_track: Her track için maksimum frame geçmişi.
        """
        self.max_track_age = max_track_age
        self.max_history_per_track = max_history_per_track
        self._tracks: dict[int, TrackState] = {}

    def update(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        frame_idx: int,
        class_name: str = "unknown",
        confidence: float = 1.0,
    ) -> TrackState:
        """Bir track'i günceller veya yeni oluşturur.

        Args:
            track_id: ByteTrack tarafından atanan takip kimliği.
            bbox: (x1, y1, x2, y2) bounding box.
            frame_idx: Mevcut frame numarası.
            class_name: Sınıf ismi.
            confidence: Tespit güven skoru.

        Returns:
            Güncellenen TrackState.
        """
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                class_name=class_name,
                max_history=self.max_history_per_track,
            )

        state = self._tracks[track_id]
        obs = BBoxPoint(
            frame_idx=frame_idx,
            x1=bbox[0], y1=bbox[1],
            x2=bbox[2], y2=bbox[3],
            confidence=confidence,
        )
        state.add_observation(obs)
        return state

    def get(self, track_id: int) -> Optional[TrackState]:
        """Track durumunu döndürür."""
        return self._tracks.get(track_id)

    def get_all_active(self, current_frame: int) -> list[TrackState]:
        """Aktif (stale olmayan) tüm track'leri döndürür."""
        return [
            s for s in self._tracks.values()
            if not s.is_stale(current_frame, self.max_track_age)
        ]

    def cleanup_stale(self, current_frame: int) -> int:
        """Eski track'leri siler.

        Args:
            current_frame: Mevcut frame numarası.

        Returns:
            Silinen track sayısı.
        """
        stale_ids = [
            tid for tid, state in self._tracks.items()
            if state.is_stale(current_frame, self.max_track_age)
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        if stale_ids:
            logger.debug(f"Temporal: {len(stale_ids)} eski track silindi")
        return len(stale_ids)

    @property
    def active_count(self) -> int:
        """Toplam aktif track sayısı."""
        return len(self._tracks)


# ─── Anomali Tespiti ──────────────────────────────────────────────────────────


@dataclass
class AnomalyResult:
    """Temporal anomali tespiti sonucu."""

    track_id: int
    is_anomalous: bool = False
    sudden_stop: bool = False
    sudden_acceleration: bool = False
    erratic_motion: bool = False
    direction_change: bool = False
    anomaly_score: float = 0.0  # 0.0 – 1.0
    details: str = ""


class TemporalAnomalyDetector:
    """Hız ve yön zaman serisinden anlık anomali tespiti.

    Algoritmalar:
        - Ani dur/kalk: Hız farkı > mean + 3σ
        - Düzensiz hareket: Hız stddev > eşik
        - Yön değişimi: Ardışık açı farkı > 90°

    Kullanım:
        detector = TemporalAnomalyDetector()
        result = detector.check(track_state)
        if result.is_anomalous:
            print(f"Anomali: {result.details}")
    """

    def __init__(
        self,
        speed_jump_sigma: float = 3.0,
        min_speed_for_stop: float = 0.5,
        erratic_std_threshold: float = 8.0,
        direction_change_deg: float = 90.0,
        min_history: int = 8,
    ) -> None:
        """
        Args:
            speed_jump_sigma: Kaç sigma üstü hız artışı "ani ivme" sayılır.
            min_speed_for_stop: Bu hızın altı "dur" sayılır (px/frame).
            erratic_std_threshold: Bu stddev üstü "düzensiz hareket" sayılır.
            direction_change_deg: Bu açı üstü "ani yön değişimi" sayılır.
            min_history: Anomali tespiti için minimum geçmiş frame sayısı.
        """
        self.speed_jump_sigma = speed_jump_sigma
        self.min_speed_for_stop = min_speed_for_stop
        self.erratic_std_threshold = erratic_std_threshold
        self.direction_change_deg = direction_change_deg
        self.min_history = min_history

    def check(self, state: TrackState) -> AnomalyResult:
        """Bir TrackState üzerinde anomali kontrolü yapar.

        Args:
            state: Kontrol edilecek track durumu.

        Returns:
            AnomalyResult — anomali bilgileri.
        """
        result = AnomalyResult(track_id=state.track_id)

        if state.age < self.min_history:
            return result

        speeds = list(state.speed_history)
        mean_speed = float(np.mean(speeds))
        std_speed = float(np.std(speeds)) if len(speeds) > 1 else 0.0
        current_speed = speeds[-1] if speeds else 0.0
        prev_speed = speeds[-2] if len(speeds) > 1 else 0.0

        anomaly_score = 0.0
        details_parts = []

        # ── 1. Ani durma ──────────────────────────────────────────────────────
        if prev_speed > mean_speed + std_speed and current_speed < self.min_speed_for_stop:
            result.sudden_stop = True
            anomaly_score += 0.4
            details_parts.append(
                f"Ani durma: {prev_speed:.1f}→{current_speed:.1f} px/f"
            )

        # ── 2. Ani ivmelenme ──────────────────────────────────────────────────
        threshold = mean_speed + self.speed_jump_sigma * (std_speed + 1e-6)
        if current_speed > threshold and current_speed > 5.0:
            result.sudden_acceleration = True
            anomaly_score += 0.3
            details_parts.append(
                f"Ani ivme: {current_speed:.1f} px/f (eşik: {threshold:.1f})"
            )

        # ── 3. Düzensiz hareket ───────────────────────────────────────────────
        if std_speed > self.erratic_std_threshold:
            result.erratic_motion = True
            anomaly_score += 0.2
            details_parts.append(
                f"Düzensiz: std={std_speed:.1f} px/f"
            )

        # ── 4. Yön değişimi ───────────────────────────────────────────────────
        points = state.trajectory_points
        if len(points) >= 3:
            direction_change = self._detect_direction_change(points[-3:])
            if direction_change:
                result.direction_change = True
                anomaly_score += 0.1
                details_parts.append("Ani yön değişimi")

        result.anomaly_score = min(1.0, anomaly_score)
        result.is_anomalous = (
            result.sudden_stop
            or result.sudden_acceleration
            or result.erratic_motion
            or result.direction_change
        )
        result.details = " | ".join(details_parts) if details_parts else ""

        # TrackState'e sonuçları yaz
        state.is_sudden_stop = result.sudden_stop
        state.is_sudden_acceleration = result.sudden_acceleration
        state.is_erratic_motion = result.erratic_motion

        return result

    def _detect_direction_change(
        self, points: list[tuple[float, float]]
    ) -> bool:
        """Üç nokta arasındaki açı değişimini kontrol eder.

        Args:
            points: Son 3 yörünge noktası [(x0,y0), (x1,y1), (x2,y2)].

        Returns:
            True: Ani yön değişimi var.
        """
        if len(points) < 3:
            return False

        p0, p1, p2 = points[-3], points[-2], points[-1]

        # İki vektör
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])

        # Büyüklükler
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        # Hareket yok
        if mag1 < 1e-6 or mag2 < 1e-6:
            return False

        # Dot product → açı
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        angle_deg = math.degrees(math.acos(cos_theta))

        return angle_deg > self.direction_change_deg


# ─── EAR/MAR Temporal Analizi ─────────────────────────────────────────────────


class DrowsinessTemporalAnalyzer:
    """EAR zaman serisinden temporal uyuklama tespiti.

    MediaPipe'ten gelen her frame EAR değerini biriktirir.
    PERCLOS (% of time eyes are < threshold) metriğini hesaplar.

    PERCLOS > %70 → uyuklama riski yüksek

    Kullanım:
        analyzer = DrowsinessTemporalAnalyzer()
        analyzer.add(ear=0.22)
        result = analyzer.is_drowsy()
    """

    EAR_THRESHOLD = 0.25
    PERCLOS_THRESHOLD = 0.70

    def __init__(
        self,
        window_size: int = 90,  # 3 saniye (30 FPS)
        ear_threshold: float = 0.25,
        perclos_threshold: float = 0.70,
    ) -> None:
        """
        Args:
            window_size: Analiz penceresi (frame sayısı).
            ear_threshold: Göz "kapalı" sayılan EAR değeri.
            perclos_threshold: PERCLOS eşiği (uyuklama sınırı).
        """
        self.window_size = window_size
        self.ear_threshold = ear_threshold
        self.perclos_threshold = perclos_threshold
        self._ear_history: deque[float] = deque(maxlen=window_size)

    def add(self, ear: float) -> None:
        """Yeni EAR değeri ekler."""
        self._ear_history.append(ear)

    def perclos(self) -> float:
        """PERCLOS oranını hesaplar (0.0 – 1.0).

        Returns:
            Gözlerin eşiğin altında kaldığı frame oranı.
        """
        if not self._ear_history:
            return 0.0
        closed = sum(1 for e in self._ear_history if e < self.ear_threshold)
        return closed / len(self._ear_history)

    def is_drowsy(self) -> bool:
        """PERCLOS eşiğini aşıyor mu?"""
        return self.perclos() > self.perclos_threshold

    def mean_ear(self) -> float:
        """Penceredeki ortalama EAR."""
        if not self._ear_history:
            return 1.0  # Varsayılan: gözler açık
        return float(np.mean(self._ear_history))

    def reset(self) -> None:
        """Geçmişi temizler (görev değişiminde)."""
        self._ear_history.clear()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO)

    # TrajectoryBuffer testi
    buf = TrajectoryBuffer(max_track_age=10)
    detector = TemporalAnomalyDetector(min_history=5)

    # Simüle edilmiş 30 frame takip
    for frame in range(30):
        x = 50 + frame * 4 + random.randint(-2, 2)
        y = 100 + random.randint(-3, 3)
        buf.update(track_id=1, bbox=(x, y, x + 60, y + 40),
                   frame_idx=frame, class_name="otomobil")

    state = buf.get(1)
    if state:
        result = detector.check(state)
        print(f"Track 1 — Anomali: {result.is_anomalous}, Skor: {result.anomaly_score:.2f}")
        print(f"Ortalama hız: {state.mean_speed:.2f} px/frame")

    # DrowsinessTemporalAnalyzer testi
    drowsy = DrowsinessTemporalAnalyzer(window_size=30)
    for i in range(30):
        # Simüle: ilk 20 frame gözler kapalı
        ear_val = 0.18 if i < 20 else 0.32
        drowsy.add(ear_val)

    print(f"PERCLOS: {drowsy.perclos():.2%}, Uyuklama: {drowsy.is_drowsy()}")
