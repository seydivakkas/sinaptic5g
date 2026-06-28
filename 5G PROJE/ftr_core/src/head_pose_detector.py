# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
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

"""F3 Task: Head Pose Detector using MediaPipe FaceMesh.

Estimates head orientation (yaw, pitch, roll) to detect distraction (etrafa_bakinma).
"""

import cv2
import numpy as np
from collections import deque

class HeadPoseDetector:
    """Estimates head yaw, pitch, roll from FaceMesh landmarks."""

    # 3D Model points for 6 canonical facial points (nose tip, chin, left eye left corner, etc.)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (1)
        (0.0, -330.0, -65.0),        # Chin (152)
        (-225.0, 170.0, -135.0),     # Left eye left corner (33)
        (225.0, 170.0, -135.0),      # Right eye right corner (263)
        (-150.0, -150.0, -125.0),    # Left mouth corner (61)
        (150.0, -150.0, -125.0)      # Right mouth corner (291)
    ], dtype=np.float64)

    # Matching FaceMesh landmark indices
    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(self, buffer_size: int = 8, trigger_count: int = 5):
        self.buffer_size = buffer_size
        self.trigger_count = trigger_count
        self.history = deque(maxlen=buffer_size)

    def process_landmarks(self, landmarks, width: int, height: int) -> dict:
        """Calculate yaw, pitch, roll and check if look-around distraction occurs.

        Returns:
            dict containing yaw, pitch, roll, is_distracted (instant), and behavior output.
        """
        image_points = []
        for idx in self.LANDMARK_INDICES:
            lm = landmarks.landmark[idx]
            image_points.append((lm.x * width, lm.y * height))
        image_points = np.array(image_points, dtype=np.float64)

        # Approximate focal length and camera center
        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "is_distracted": False, "behavior": None}

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Extract Euler angles (pitch, yaw, roll)
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0.0

        # Angles in degrees
        pitch = float(np.degrees(x))
        yaw = float(np.degrees(y))
        roll = float(np.degrees(z))

        # Check conditions: |yaw| > 30 degrees OR |pitch| > 20 degrees
        instant_distracted = abs(yaw) > 30.0 or abs(pitch) > 20.0
        self.history.append(instant_distracted)

        # Buffer condition: trigger if at least trigger_count frames in sliding window are distracted
        is_triggered = sum(self.history) >= self.trigger_count

        result = {
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
            "is_distracted": instant_distracted,
            "behavior": {"etiket": "etrafa_bakinma", "confidence_score": 0.75} if is_triggered else None
        }
        return result
