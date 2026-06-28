"""
driver_analyzer.py — Sinaptic5G MediaPipe Sürücü Davranış Analizi
=================================================================
MediaPipe FaceMesh ve Hands ile sürücünün uyuklama, esneme,
telefon ve sigara kullanım durumunu gerçek zamanlı tespit eder.

EAR (Eye Aspect Ratio) — Uyuklama Tespiti:
    EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 × ‖p1-p4‖)
    EAR < 0.25 → Göz kapandı (uyuklama riski)

MAR (Mouth Aspect Ratio) — Esneme Tespiti:
    MAR = ‖dikey_uzaklık‖ / ‖yatay_uzaklık‖
    MAR > 0.60 → Ağız açık (esneme)
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    mp_hands    = mp.solutions.hands
    mp_face     = mp.solutions.face_mesh
    mp_drawing  = mp.solutions.drawing_utils
    mp_styles   = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    mp_hands = None
    mp_face = None
    mp_drawing = None
    mp_styles = None
    logger = logging.getLogger(__name__) # Keep logger defined
    logger.warning("MediaPipe is not installed. Driver analysis features will be disabled.")


@dataclass
class DriverState:
    """Sürücünün anlık durum raporu."""
    is_drowsy:        bool  = False    # Uyuklama
    is_yawning:       bool  = False    # Esneme
    is_holding_phone: bool  = False    # Telefon tutuyor mu?
    is_smoking:       bool  = False    # Sigara içiyor mu?
    ear_value:        float = 0.0      # Göz kısılma oranı
    mar_value:        float = 0.0      # Ağız açıklık oranı
    drowsy_frame_count: int = 0        # Kaç kare uyukluyor
    face_detected:    bool  = False    # Yüz algılandı mı?
    hands_detected:   int   = 0        # Kaç el algılandı

    @property
    def is_distracted(self) -> bool:
        """Sürücü dikkatini dağıtıyor mu?"""
        return (
            self.is_drowsy or
            self.is_yawning or
            self.is_holding_phone or
            self.is_smoking
        )


class DriverAnalyzer:
    """
    MediaPipe tabanlı sürücü davranış analizi.

    Analiz edilen davranışlar:
    - EAR < 0.25 için 30 kare (%1 sn @30fps) → uyuklama uyarısı
    - MAR > 0.60 → esneme tespiti
    - El yüz seviyesindeyse + telefon tespiti → telefon kullanımı
    - El yüz seviyesindeyse + sigara tespiti → sigara kullanımı

    Kişiselleştirme:
    - İlk 300 kare kalibrasyon fazı ile sürücüye özgü EAR eşiği hesaplanabilir
    """

    # MediaPipe FaceMesh landmark indeksleri
    # Sol göz: dış köşe(33), üst(160,158), iç köşe(133), alt(153,144)
    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    # Ağız landmark indeksleri
    MOUTH_IDX = [13, 14, 312, 82, 311, 87]

    # Eşikler
    EAR_THRESHOLD  = 0.25
    MAR_THRESHOLD  = 0.60
    DROWSY_FRAMES  = 30    # ~1 saniye (30 FPS'te)
    DROWSY_PENALTY = 2     # EAR normale döndüğünde azalış hızı

    # El → yüz koordinat eşiği
    FACE_LEVEL = 0.35      # Görüntünün üst %35'i

    # Kalibrasyon
    CALIBRATION_FRAMES = 300

    def __init__(
        self,
        ear_threshold: float = None,     # None → varsayılan kullanılır
        calibration_mode: bool = False,  # True → kişiye özel EAR kalibre eder
    ):
        if MEDIAPIPE_AVAILABLE:
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            self.face_mesh = mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.hands = None
            self.face_mesh = None

        self._ear_threshold = ear_threshold or self.EAR_THRESHOLD
        self._drowsy_counter = 0
        self._ear_history   = deque(maxlen=10)   # EMA için
        self._ear_calibration: list[float] = []
        self._calibration_mode = calibration_mode

    def analyze(
        self,
        frame: np.ndarray,
        has_phone_detected: bool = False,
        has_cigarette_detected: bool = False,
    ) -> DriverState:
        """
        Kareyi analiz eder ve DriverState döner.

        YOLO tespitlerini (has_phone, has_cigarette) bağlamsal bilgi
        olarak kullanır — sadece el-yüz konumundan değil, nesne
        tespitinden de faydalanır.

        Parametreler:
            frame: BGR numpy array (kamera karesi)
            has_phone_detected: YOLO telefon tespiti yaptı mı?
            has_cigarette_detected: YOLO sigara tespiti yaptı mı?

        Dönüş:
            DriverState dataclass nesnesi
        """
        state = DriverState()
        
        if not MEDIAPIPE_AVAILABLE:
            return state

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ─── Yüz Analizi (FaceMesh) ───────────────────────────────────────
        face_results = self.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            state.face_detected = True
            landmarks = face_results.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            ear = self._compute_ear(landmarks, w, h)
            mar = self._compute_mar(landmarks, w, h)

            state.ear_value = round(ear, 3)
            state.mar_value = round(mar, 3)

            # Kalibrasyon fazı
            if self._calibration_mode:
                self._ear_calibration.append(ear)
                if len(self._ear_calibration) >= self.CALIBRATION_FRAMES:
                    self._calibrate_ear()

            # EMA ile yumuşatma (gürültüyü azaltır)
            self._ear_history.append(ear)
            smooth_ear = sum(self._ear_history) / len(self._ear_history)

            # Uyuklama sayacı güncelle
            if smooth_ear < self._ear_threshold:
                self._drowsy_counter += 1
            else:
                # Normale döndüğünde yavaş azalt (histerezis)
                self._drowsy_counter = max(
                    0, self._drowsy_counter - self.DROWSY_PENALTY
                )

            state.drowsy_frame_count = self._drowsy_counter
            state.is_drowsy  = self._drowsy_counter >= self.DROWSY_FRAMES
            state.is_yawning = mar > self.MAR_THRESHOLD

            logger.debug(
                f"EAR={ear:.3f}(smooth:{smooth_ear:.3f}) "
                f"MAR={mar:.3f} Drowsy={self._drowsy_counter}"
            )

        # ─── El Analizi (Hands) ───────────────────────────────────────────
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            state.hands_detected = len(hand_results.multi_hand_landmarks)

            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Bileğin normalize y koordinatı (0=üst, 1=alt)
                wrist_y = hand_landmarks.landmark[0].y

                # El yüz seviyesindeyse (görüntünün üst bölgesi) → kontrol et
                if wrist_y < self.FACE_LEVEL:
                    if has_phone_detected:
                        state.is_holding_phone = True
                        logger.debug("Telefon kullanımı tespit edildi")

                    if has_cigarette_detected:
                        state.is_smoking = True
                        logger.debug("Sigara kullanımı tespit edildi")

        return state

    def _compute_ear(
        self,
        landmarks,
        width: int,
        height: int,
    ) -> float:
        """
        Eye Aspect Ratio (EAR) hesaplar.

        Formül:
            EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 × ‖p1-p4‖)

        Her iki göz için ayrı hesap yapılır, ortalaması alınır.
        """
        def lp(idx: int) -> np.ndarray:
            lm = landmarks.landmark[idx]
            return np.array([lm.x * width, lm.y * height])

        def ear_single(eye_indices: list[int]) -> float:
            points = [lp(i) for i in eye_indices]
            # Dikey mesafeler
            A = np.linalg.norm(points[1] - points[5])
            B = np.linalg.norm(points[2] - points[4])
            # Yatay mesafe
            C = np.linalg.norm(points[0] - points[3])
            return (A + B) / (2.0 * C + 1e-6)

        left_ear  = ear_single(self.LEFT_EYE_IDX)
        right_ear = ear_single(self.RIGHT_EYE_IDX)
        return (left_ear + right_ear) / 2.0

    def _compute_mar(
        self,
        landmarks,
        width: int,
        height: int,
    ) -> float:
        """
        Mouth Aspect Ratio (MAR) hesaplar.

        Formül:
            MAR = ‖dikey_uzaklık‖ / ‖yatay_uzaklık‖
        """
        def lp(idx: int) -> np.ndarray:
            lm = landmarks.landmark[idx]
            return np.array([lm.x * width, lm.y * height])

        mouth = [lp(i) for i in self.MOUTH_IDX]
        vertical   = np.linalg.norm(mouth[0] - mouth[1])
        horizontal = np.linalg.norm(mouth[2] - mouth[3])
        return vertical / (horizontal + 1e-6)

    def _calibrate_ear(self):
        """
        Toplanan EAR değerlerinden kişiye özel eşik hesaplar.
        Ortalama - 2 standart sapma ile eşik belirlenir.
        """
        if len(self._ear_calibration) < 50:
            return

        arr = np.array(self._ear_calibration)
        mean_ear = float(np.mean(arr))
        std_ear  = float(np.std(arr))

        # Kişiselleştirilmiş eşik
        calibrated = mean_ear - 2 * std_ear
        calibrated = max(0.15, min(0.30, calibrated))  # 0.15-0.30 arası kısıtla

        self._ear_threshold = calibrated
        self._calibration_mode = False  # Kalibrasyon tamamlandı

        logger.info(
            f"EAR kalibrasyonu tamamlandı: "
            f"mean={mean_ear:.3f}, std={std_ear:.3f} → "
            f"eşik={self._ear_threshold:.3f}"
        )

    def draw_face_mesh(self, frame: np.ndarray) -> np.ndarray:
        """
        YüZ mesh'ini kare üzerine çizer (debug amaçlı).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb)
        annotated = frame.copy()

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    face_landmarks,
                    mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                )

        return annotated

    def reset(self):
        """Uyuklama sayacını sıfırlar."""
        self._drowsy_counter = 0
        self._ear_history.clear()

    def close(self):
        """MediaPipe kaynaklarını serbest bırakır."""
        if self.hands:
            self.hands.close()
        if self.face_mesh:
            self.face_mesh.close()


if __name__ == "__main__":
    print("DriverAnalyzer modülü hazır.")
    print(f"EAR eşiği: {DriverAnalyzer.EAR_THRESHOLD}")
    print(f"MAR eşiği: {DriverAnalyzer.MAR_THRESHOLD}")
    print(f"Uyuklama kare sayısı: {DriverAnalyzer.DROWSY_FRAMES}")
