"""
behavior_classifier.py — Sinaptic5G Sürücü Davranış Temporal Sınıflandırıcı
=============================================================================
Çoklu modalite sinyallerini (EAR, MAR, hız, yön) birleştirerek
sürücünün davranışını temporal pencerede sınıflandırır.

Sınıflar:
    DrivingBehaviorState — Anlık davranış durumu dataclass
    BehaviorClassifier   — Kuralı + olasılıksal birleştirme

Davranış Kategorileri:
    NORMAL      : Risk yok
    DISTRACTED  : Dikkat dağıtıcı nesne (telefon/sigara)
    DROWSY      : Uyuklama/esneme zinciri
    AGGRESSIVE  : Ani hız değişimleri
    CRITICAL    : Birden fazla risk faktörü aynı anda

Kullanım:
    from src.models.temporal.behavior_classifier import BehaviorClassifier

    clf = BehaviorClassifier()
    state = clf.classify(
        ear=0.21, mar=0.35,
        has_phone=True, has_cigarette=False,
        speed_kmh=72.0, speed_limit=50.0,
        is_erratic=False
    )
    print(state.category, state.confidence)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.models.temporal.trajectory import DrowsinessTemporalAnalyzer

logger = logging.getLogger(__name__)


class BehaviorCategory(str, Enum):
    """Sürücü davranış kategorisi."""
    NORMAL = "NORMAL"
    DISTRACTED = "DISTRACTED"
    DROWSY = "DROWSY"
    AGGRESSIVE = "AGGRESSIVE"
    CRITICAL = "CRITICAL"


@dataclass
class DrivingBehaviorState:
    """Anlık sürücü davranış durumu.

    Attributes:
        category: Davranış kategorisi.
        confidence: Sınıflandırma güveni (0.0 – 1.0).
        risk_score: Birleşik risk skoru (0 – 100).
        requires_attention: Yerel uyarı arayüzü dikkat göstermeli mi?
        evidence: Hangi sinyaller aktif (debug için).
    """

    category: BehaviorCategory = BehaviorCategory.NORMAL
    confidence: float = 1.0
    risk_score: float = 0.0
    requires_attention: bool = False
    evidence: list[str] = None  # type: ignore

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class BehaviorClassifier:
    """Temporal sürücü davranış sınıflandırıcısı.

    Kural tabanlı ağırlıklı risk skoru ve eşik sistemiyle
    davranış kategorisini belirler. PERCLOS (temporal EAR)
    entegrasyonu ile anlık EAR kararlarından daha güvenilir
    uyuklama tespiti sağlar.

    Risk Ağırlıkları:
        Telefon (aktif)        → 35 puan
        Uyuklama (PERCLOS)     → 30 puan
        Hız ihlali (>20km/s)  → 25 puan
        Sigara                 → 15 puan
        Esneme (MAR>0.6)       → 8 puan
        Ani hız değişimi       → 10 puan

    Eşikler:
        risk < 25  → NORMAL
        risk < 50  → DISTRACTED/DROWSY (uyarı)
        risk < 75  → AGGRESSIVE
        risk ≥ 75  → CRITICAL yerel uyarı

    Kullanım:
        clf = BehaviorClassifier()
        state = clf.classify(ear=0.22, mar=0.35, ...)
    """

    # Risk puan ağırlıkları
    WEIGHTS = {
        "phone": 35,
        "drowsy_perclos": 30,
        "speed_excess": 25,
        "cigarette": 15,
        "erratic_motion": 10,
        "yawning": 8,
        "drowsy_instant": 5,  # Anlık EAR < eşik (PERCLOS olmadan)
    }

    # Eşikler
    THRESHOLD_WARNING = 25.0
    THRESHOLD_AGGRESSIVE = 50.0
    THRESHOLD_CRITICAL = 75.0

    # Yalnız yerel UI uyarısı; QoD kararı değildir.
    ATTENTION_SCORE = 50.0

    def __init__(
        self,
        ear_threshold: float = 0.25,
        mar_threshold: float = 0.60,
        perclos_window: int = 90,
        speed_limit_kmh: float = 50.0,
        speed_excess_threshold_kmh: float = 20.0,
    ) -> None:
        """
        Args:
            ear_threshold: EAR uyuklama eşiği.
            mar_threshold: MAR esneme eşiği.
            perclos_window: PERCLOS hesabı için frame penceresi.
            speed_limit_kmh: Varsayılan hız limiti.
            speed_excess_threshold_kmh: Bu kadar km/s fazla → risk aktif.
        """
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.speed_limit_kmh = speed_limit_kmh
        self.speed_excess_threshold_kmh = speed_excess_threshold_kmh

        # PERCLOS penceresi
        self.drowsiness_analyzer = DrowsinessTemporalAnalyzer(
            window_size=perclos_window,
            ear_threshold=ear_threshold,
        )

    def classify(
        self,
        ear: float,
        mar: float,
        has_phone: bool = False,
        has_cigarette: bool = False,
        speed_kmh: Optional[float] = None,
        speed_limit_kmh: Optional[float] = None,
        is_erratic: bool = False,
        phone_confidence: float = 0.0,
        cigarette_confidence: float = 0.0,
    ) -> DrivingBehaviorState:
        """Sürücü davranışını sınıflandırır.

        Args:
            ear: Göz kısılma oranı (Eye Aspect Ratio).
            mar: Ağız açıklık oranı (Mouth Aspect Ratio).
            has_phone: Telefon tespiti var mı?
            has_cigarette: Sigara tespiti var mı?
            speed_kmh: Mevcut hız (km/s). None → hız bilinmiyor.
            speed_limit_kmh: Hız limiti (None → varsayılan kullanılır).
            is_erratic: Temporal anomali: düzensiz hareket.
            phone_confidence: Telefon güven skoru (0.0–1.0).
            cigarette_confidence: Sigara güven skoru.

        Returns:
            DrivingBehaviorState — sınıflandırma sonucu.
        """
        # PERCLOS güncelle
        self.drowsiness_analyzer.add(ear)

        limit = speed_limit_kmh or self.speed_limit_kmh
        risk = 0.0
        evidence: list[str] = []

        # ── 1. Telefon ────────────────────────────────────────────────────────
        if has_phone and phone_confidence > 0.5:
            risk += self.WEIGHTS["phone"]
            evidence.append(f"telefon ({phone_confidence:.2f})")

        # ── 2. Uyuklama (PERCLOS) ─────────────────────────────────────────────
        perclos = self.drowsiness_analyzer.perclos()
        if self.drowsiness_analyzer.is_drowsy():
            risk += self.WEIGHTS["drowsy_perclos"]
            evidence.append(f"uyuklama PERCLOS={perclos:.0%}")
        elif ear < self.ear_threshold:
            # Anlık EAR düşük ama PERCLOS eşiği aşılmadı
            risk += self.WEIGHTS["drowsy_instant"]
            evidence.append(f"anlık EAR={ear:.3f}")

        # ── 3. Hız ihlali ─────────────────────────────────────────────────────
        if speed_kmh is not None:
            excess = speed_kmh - limit
            if excess > self.speed_excess_threshold_kmh:
                # Lineer normalleştirme: 20 km/s fazla → tam puan
                factor = min(1.0, (excess - self.speed_excess_threshold_kmh) / 20.0)
                speed_risk = self.WEIGHTS["speed_excess"] * factor
                risk += speed_risk
                evidence.append(
                    f"hız ihlali {speed_kmh:.0f}/{limit:.0f} km/s"
                )

        # ── 4. Sigara ─────────────────────────────────────────────────────────
        if has_cigarette and cigarette_confidence > 0.5:
            risk += self.WEIGHTS["cigarette"]
            evidence.append(f"sigara ({cigarette_confidence:.2f})")

        # ── 5. Esneme ─────────────────────────────────────────────────────────
        if mar > self.mar_threshold:
            risk += self.WEIGHTS["yawning"]
            evidence.append(f"esneme MAR={mar:.2f}")

        # ── 6. Düzensiz hareket ───────────────────────────────────────────────
        if is_erratic:
            risk += self.WEIGHTS["erratic_motion"]
            evidence.append("düzensiz hareket")

        risk = min(100.0, risk)

        # ── Kategori belirleme ────────────────────────────────────────────────
        category, confidence = self._determine_category(
            risk=risk,
            has_phone=has_phone,
            is_drowsy=self.drowsiness_analyzer.is_drowsy(),
            is_erratic=is_erratic,
        )

        requires_attention = risk >= self.ATTENTION_SCORE

        if requires_attention:
            logger.info(
                f"[BehaviorClassifier] yerel dikkat uyarısı: "
                f"risk={risk:.1f}, kategori={category}, kanıt={evidence}"
            )

        return DrivingBehaviorState(
            category=category,
            confidence=confidence,
            risk_score=risk,
            requires_attention=requires_attention,
            evidence=evidence,
        )

    def _determine_category(
        self,
        risk: float,
        has_phone: bool,
        is_drowsy: bool,
        is_erratic: bool,
    ) -> tuple[BehaviorCategory, float]:
        """Risk skorundan davranış kategorisini belirler.

        Returns:
            (category, confidence) tuple'ı.
        """
        if risk >= self.THRESHOLD_CRITICAL:
            return BehaviorCategory.CRITICAL, 0.95

        if risk >= self.THRESHOLD_AGGRESSIVE:
            if is_erratic:
                return BehaviorCategory.AGGRESSIVE, 0.85
            if has_phone or is_drowsy:
                return BehaviorCategory.DISTRACTED, 0.80
            return BehaviorCategory.AGGRESSIVE, 0.75

        if risk >= self.THRESHOLD_WARNING:
            if is_drowsy:
                return BehaviorCategory.DROWSY, 0.80
            if has_phone:
                return BehaviorCategory.DISTRACTED, 0.75
            return BehaviorCategory.DISTRACTED, 0.60

        return BehaviorCategory.NORMAL, 0.95

    def reset(self) -> None:
        """Temporal analiz geçmişini sıfırlar (oturum değişiminde)."""
        self.drowsiness_analyzer.reset()
        logger.debug("BehaviorClassifier geçmişi sıfırlandı")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    clf = BehaviorClassifier(speed_limit_kmh=50.0)

    # Senaryo 1: Normal sürüş
    s1 = clf.classify(ear=0.32, mar=0.20, has_phone=False, speed_kmh=45.0)
    print(f"Senaryo 1 — {s1.category}: risk={s1.risk_score:.1f}, dikkat={s1.requires_attention}")

    # Senaryo 2: Telefon + hız ihlali
    s2 = clf.classify(
        ear=0.30, mar=0.22, has_phone=True, phone_confidence=0.85,
        speed_kmh=78.0
    )
    print(f"Senaryo 2 — {s2.category}: risk={s2.risk_score:.1f}, dikkat={s2.requires_attention}")
    print(f"  Kanıt: {s2.evidence}")

    # Senaryo 3: PERCLOS uyuklama (90 frame uyuyor)
    for _ in range(90):
        clf.drowsiness_analyzer.add(0.18)  # Gözler kapalı
    s3 = clf.classify(ear=0.18, mar=0.25)
    print(f"Senaryo 3 — {s3.category}: risk={s3.risk_score:.1f}, dikkat={s3.requires_attention}")
    print(f"  PERCLOS: {clf.drowsiness_analyzer.perclos():.0%}")
