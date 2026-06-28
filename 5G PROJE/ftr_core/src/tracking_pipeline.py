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

import json
import os
import logging
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import cv2
import numpy as np

# Hungarian matching — scipy ile mevcut değilse greedy fallback aktif
try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

logger = logging.getLogger("sinaptic5g.tracking")

# ══════════════════════════════════════════════════════════════════
# FAZ 1: KAMERA KALİBRASYONU (İç Parametre + BEV Homografisi)
# ══════════════════════════════════════════════════════════════════

class CameraCalibrator:
    """Intrinsics and bird's eye view perspective transform manager."""

    CALIB_FILE = Path("configs/camera_calibration.json")

    @staticmethod
    def load(frame_wh: Tuple[int, int] = (1280, 720)) -> Tuple[np.ndarray, np.ndarray]:
        """Loads calibration parameters, falls back if not found."""
        if CameraCalibrator.CALIB_FILE.is_file():
            try:
                with open(CameraCalibrator.CALIB_FILE, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                K = np.array(data["K"], dtype=np.float64)
                D = np.array(data["D"], dtype=np.float64)
                return K, D
            except Exception:
                pass
        
        # Fallback intrinsics estimation
        w, h = frame_wh
        K = np.array([
            [w * 0.9, 0.0,    w / 2.0],
            [0.0,     w * 0.9, h / 2.0],
            [0.0,     0.0,     1.0    ]
        ], dtype=np.float64)
        D = np.zeros((1, 5), dtype=np.float64)
        return K, D

    @staticmethod
    def compute_bev_homography(camera_mode: str, frame_wh: Tuple[int, int] = (1280, 720)) -> np.ndarray:
        """Calculates perspective homography mapping road plane to metric coordinates."""
        w, h = frame_wh

        points = {
            "front": {
                "src": np.float32([
                    [w * 0.25, h * 0.72],
                    [w * 0.75, h * 0.72],
                    [w * 0.55, h * 0.55],
                    [w * 0.45, h * 0.55],
                ]),
                "dst": np.float32([
                    [0.0,  0.0],
                    [3.5,  0.0],
                    [0.0, 25.0],
                    [3.5, 25.0],
                ])
            },
            "side": {
                "src": np.float32([
                    [w * 0.08, h * 0.75],
                    [w * 0.92, h * 0.75],
                    [w * 0.08, h * 0.50],
                    [w * 0.92, h * 0.50],
                ]),
                "dst": np.float32([
                    [0.0,  0.0],
                    [40.0, 0.0],
                    [0.0,  4.5],
                    [40.0, 4.5],
                ])
            }
        }

        pts = points.get(camera_mode, points["front"])
        H, _ = cv2.findHomography(pts["src"], pts["dst"], cv2.RANSAC, 5.0)
        return H if H is not None else np.eye(3, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# FAZ 3: REID EMBEDDING (ONNX osnet_x0_25 with HSV Fallback)
# ══════════════════════════════════════════════════════════════════

class BoTSORTReID:
    """ReID extractor wrapper for BoTSORT using osnet_x0_25.onnx, falls back to HSV signature."""

    def __init__(self, model_path: str = "models/osnet_x0_25.onnx", bins: int = 32) -> None:
        self.session = None
        self.bins = bins
        self.gallery: Dict[int, np.ndarray] = {}

        resolved_path = Path(model_path)
        if not resolved_path.is_file():
            resolved_path = Path("5G PROJE") / model_path
            if not resolved_path.is_file():
                resolved_path = Path("/app") / model_path

        if resolved_path.is_file():
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
                self.session = ort.InferenceSession(str(resolved_path), providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                logger.info("Loaded osnet_x0_25 ReID model successfully.")
            except Exception as e:
                logger.error("Failed to load osnet_x0_25 ONNX session: %s", e)

    def extract_signature(self, crop: np.ndarray) -> np.ndarray:
        """Extracts feature vector from image crop using ONNX model or HSV fallback."""
        if crop is None or crop.size == 0:
            return np.zeros(self.bins * 2 if self.session is None else 512, dtype=np.float32)

        if self.session is not None:
            try:
                # Preprocess for OSNet: typical input is 256x128 float32 normalized image
                resized = cv2.resize(crop, (128, 256))
                blob = resized.astype(np.float32) / 255.0
                # Mean & std normalization for OSNet
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                blob = (blob - mean) / std
                blob = np.transpose(blob, (2, 0, 1)).astype(np.float32)
                blob = np.expand_dims(blob, axis=0)

                feat = self.session.run(None, {self.input_name: blob})[0][0]
                norm = np.linalg.norm(feat)
                return feat / (norm + 1e-6)
            except Exception as e:
                logger.debug("OSNet inference failed, falling back to HSV: %s", e)

        # Fallback: HSV histogram signature
        h, w = crop.shape[:2]
        roi = crop[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        if roi.size == 0:
            return np.zeros(self.bins * 2, dtype=np.float32)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [self.bins], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256]).flatten()
        feat = np.concatenate([h_hist, s_hist])
        norm = feat.sum()
        return (feat / (norm + 1e-6)).astype(np.float32)

    def register(self, track_id: int, crop: np.ndarray) -> None:
        if track_id not in self.gallery:
            self.gallery[track_id] = self.extract_signature(crop)

    def similarity(self, track_id: int, crop: np.ndarray) -> float:
        if track_id not in self.gallery:
            return 1.0
        current = self.extract_signature(crop)
        stored = self.gallery[track_id]
        if self.session is not None:
            # Cosine similarity for OSNet features
            return float(np.dot(stored, current))
        # Bhattacharyya distance converted to similarity for HSV
        dist = cv2.compareHist(stored, current, cv2.HISTCMP_BHATTACHARYYA)
        return float(1.0 - dist)

    def verify_id(self, track_id: int, crop: np.ndarray, threshold: float = 0.55) -> bool:
        sim = self.similarity(track_id, crop)
        if sim < threshold:
            self.gallery[track_id] = self.extract_signature(crop)
            return False
        return True

    def cleanup(self, active_ids: Set[int]) -> None:
        for stale_id in [k for k in self.gallery if k not in active_ids]:
            self.gallery.pop(stale_id, None)


# ══════════════════════════════════════════════════════════════════
# FAZ 4: DİNAMİK KALMAN FİLTRESİ
# ══════════════════════════════════════════════════════════════════

class AdaptiveKalmanTracker:
    """BEV coordinate state estimation Kalman filter, dynamically adapted to detection confidence."""

    def __init__(
        self,
        track_id: int,
        init_bev: np.ndarray,
        camera_mode: str = "front",
        init_timestamp: float = 0.0,
    ) -> None:
        self.track_id = track_id
        self.camera_mode = camera_mode
        self.age = 0
        self.lost_count = 0
        self.history: deque = deque(maxlen=30)
        self.time_history: deque = deque(maxlen=30)
        self.history.append(init_bev.copy())
        self.time_history.append(float(init_timestamp))
        self.last_timestamp = float(init_timestamp)

        # State vector: [bev_x, bev_y, bev_vx, bev_vy]
        # Measurement vector: [bev_x, bev_y]
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0 / 25.0
        
        self.kf.transitionMatrix = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        self._init_noise_matrices(camera_mode)
        self.kf.statePre = np.array([init_bev[0], init_bev[1], 0.0, 0.0], dtype=np.float32).reshape(4, 1)
        self.kf.statePost = self.kf.statePre.copy()

    def _init_noise_matrices(self, mode: str) -> None:
        if mode == "front":
            q_pos = 0.05
            q_vel = 0.80
            r_val = 0.30
        elif mode == "side":
            q_pos = 0.10
            q_vel = 1.50
            r_val = 0.60
        else:
            q_pos, q_vel, r_val = 0.07, 1.00, 0.40

        self.kf.processNoiseCov = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_val
        self.kf.errorCovPre = np.eye(4, dtype=np.float32) * 10.0

    def adapt_r_to_confidence(self, confidence: float) -> None:
        base_r = 0.30 if self.camera_mode == "front" else 0.60
        noise_factor = 1.0 + 3.0 * (1.0 - max(confidence, 0.30))
        r_val = base_r * noise_factor
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_val

    def predict(self, timestamp: float | None = None) -> np.ndarray:
        if timestamp is not None:
            dt = max(1e-3, min(1.0, float(timestamp) - self.last_timestamp))
            self.kf.transitionMatrix[0, 2] = dt
            self.kf.transitionMatrix[1, 3] = dt
            self.last_timestamp = float(timestamp)
        pred = self.kf.predict()
        self.age += 1
        return pred[:2].flatten()

    def update(
        self,
        bev_xy: np.ndarray,
        confidence: float = 1.0,
        timestamp: float | None = None,
    ) -> np.ndarray:
        self.adapt_r_to_confidence(confidence)
        meas = bev_xy.reshape(2, 1).astype(np.float32)
        state = self.kf.correct(meas)
        self.lost_count = 0
        self.history.append(bev_xy.copy())
        current_timestamp = self.last_timestamp if timestamp is None else float(timestamp)
        self.time_history.append(current_timestamp)
        self.last_timestamp = current_timestamp
        return state[:2].flatten()

    def apply_cmc(self, dx: float, dy: float) -> None:
        """Applies Camera Motion Compensation updates to Kalman state filter positions."""
        self.kf.statePost[0, 0] += dx
        self.kf.statePost[1, 0] += dy
        self.kf.statePre[0, 0] += dx
        self.kf.statePre[1, 0] += dy

    def mark_lost(self) -> None:
        self.lost_count += 1

    @property
    def estimated_speed_kmh(self) -> float:
        """Calculates smoothed speed in km/h based on Kalman-smoothed BEV trajectory coordinates."""
        if len(self.history) < 5:
            return 0.0
        recent = list(self.history)[-5:]
        recent_times = list(self.time_history)[-5:]
        elapsed = recent_times[-1] - recent_times[0]
        if elapsed <= 1e-6:
            return 0.0
        displacements = [
            np.linalg.norm(recent[i+1] - recent[i])
            for i in range(len(recent)-1)
        ]
        mps = float(np.sum(displacements)) / elapsed
        return mps * 3.6


# ══════════════════════════════════════════════════════════════════
# FAZ 5: BoTSORT TABANLI TAKİP MOTORU
# ══════════════════════════════════════════════════════════════════

class SinapticTracker:
    """BoTSORT multi-object tracking implementation featuring two-stage association."""

    def __init__(self,
                 camera_mode: str = "front",
                 bev_H: Optional[np.ndarray] = None,
                 conf_thresh: float = 0.45,
                 track_buffer: int = 45) -> None:
        self.camera_mode = camera_mode
        self.H_bev = bev_H if bev_H is not None else np.eye(3, dtype=np.float32)
        self.conf_thresh = conf_thresh
        self.track_buffer = track_buffer

        # BoTSORT threshold limits
        self.track_high_thresh = 0.45
        self.track_low_thresh = 0.15
        self.new_track_thresh = 0.50
        self.match_thresh = 0.80

        self.trackers: Dict[int, AdaptiveKalmanTracker] = {}
        self.last_bboxes: Dict[int, List[float]] = {}
        self.reid = BoTSORTReID()
        self.next_id = 1
        
        # Camera Motion Compensation state
        self.prev_gray: Optional[np.ndarray] = None

    def _bbox_foot_to_bev(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        foot_px = np.array([[[(x1 + x2) / 2.0, y2]]], dtype=np.float32)
        bev_pt = cv2.perspectiveTransform(foot_px, self.H_bev)
        return bev_pt[0][0]

    def _iou(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / (union + 1e-6)

    def _compute_cmc(self, frame: np.ndarray) -> Tuple[float, float]:
        """Calculates camera global translation using Lucas-Kanade optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy = 0.0, 0.0
        if self.prev_gray is not None:
            try:
                # Sparse feature detection
                pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
                if pts is not None:
                    next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None)
                    good_prev = pts[status == 1]
                    good_next = next_pts[status == 1]
                    if len(good_prev) > 0:
                        diffs = good_next - good_prev
                        dx, dy = np.median(diffs[:, 0]), np.median(diffs[:, 1])
            except Exception:
                pass
        self.prev_gray = gray
        return float(dx), float(dy)

    def _build_cost_matrix(
        self,
        detections: List[Dict[str, Any]],
        track_ids: List[int],
        timestamp: float,
        use_reid: bool = True,
    ) -> np.ndarray:
        """IoU + BEV proximity + ReID skorundan maliyet matrisi oluşturur."""
        max_bev_distance = 8.0 if self.camera_mode == "side" else 4.0
        score_matrix = np.full((len(track_ids), len(detections)), -1.0, dtype=np.float32)

        for i, tid in enumerate(track_ids):
            tracker = self.trackers[tid]
            pred_bev = tracker.predict(timestamp)
            previous_bbox = self.last_bboxes.get(tid)
            for j, det in enumerate(detections):
                overlap = self._iou(previous_bbox, det["bbox"]) if previous_bbox else 0.0
                bev_distance = float(np.linalg.norm(pred_bev - det["bev"]))

                if overlap < 0.1 and bev_distance > max_bev_distance:
                    continue

                distance_score = max(0.0, 1.0 - bev_distance / max_bev_distance)
                cost = 0.5 * overlap + 0.3 * distance_score
                if use_reid:
                    reid_sim = self.reid.similarity(tid, det["crop"])
                    cost += 0.2 * reid_sim
                score_matrix[i, j] = cost

        return score_matrix

    def _hungarian_match(
        self,
        score_matrix: np.ndarray,
        track_ids: List[int],
        threshold: float,
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Hungarian (Macar) algoritması ile optimal eşleştirme.
        scipy.optimize.linear_sum_assignment kullanır.
        """
        n_tracks, n_dets = score_matrix.shape
        matched: Dict[int, int] = {}
        used_dets: Set[int] = set()
        used_tracks: Set[int] = set()

        # Geçerli (>= threshold) eşleşme adayları var mı?
        valid_mask = score_matrix >= threshold
        if not np.any(valid_mask):
            return {}, list(range(n_dets)), list(track_ids)

        # Maliyet matrisi: score → negatif (minimizasyon)
        cost_matrix = -score_matrix.copy()
        # Geçersiz hücrelere büyük ceza
        cost_matrix[~valid_mask] = 1e6

        try:
            row_ind, col_ind = _scipy_lsa(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if valid_mask[r, c] and score_matrix[r, c] >= threshold:
                    tid = track_ids[r]
                    matched[tid] = c
                    used_dets.add(c)
                    used_tracks.add(tid)
        except Exception as exc:
            logger.debug("Hungarian matching hatası: %s — greedy fallback", exc)
            return self._greedy_match(score_matrix, track_ids, threshold)

        unmatched_dets = [j for j in range(n_dets) if j not in used_dets]
        unmatched_tracks = [tid for tid in track_ids if tid not in used_tracks]
        return matched, unmatched_dets, unmatched_tracks

    def _greedy_match(
        self,
        score_matrix: np.ndarray,
        track_ids: List[int],
        threshold: float,
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        """Greedy (açgözlü) eşleştirme — scipy yoksa veya hata durumunda fallback."""
        n_tracks, n_dets = score_matrix.shape
        matched: Dict[int, int] = {}
        used_dets: Set[int] = set()
        used_tracks: Set[int] = set()
        score_copy = score_matrix.copy()

        for _ in range(min(n_tracks, n_dets)):
            i, j = np.unravel_index(np.argmax(score_copy), score_copy.shape)
            if score_copy[i, j] < threshold:
                break
            matched[track_ids[i]] = j
            used_dets.add(j)
            used_tracks.add(track_ids[i])
            score_copy[i, :] = -1.0
            score_copy[:, j] = -1.0

        unmatched_dets = [j for j in range(n_dets) if j not in used_dets]
        unmatched_tracks = [tid for tid in track_ids if tid not in used_tracks]
        return matched, unmatched_dets, unmatched_tracks

    def _associate(
        self,
        detections: List[Dict[str, Any]],
        track_ids: List[int],
        threshold: float,
        timestamp: float,
        use_reid: bool = True,
        use_hungarian: bool = True,
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Track-detection eşleştirme.
        use_hungarian=True (scipy mevcut ise): Hungarian algoritması kullanılır.
        Fallback: Greedy eşleştirme (her zaman güvenli).
        """
        if not track_ids or not detections:
            return {}, list(range(len(detections))), list(track_ids)

        score_matrix = self._build_cost_matrix(detections, track_ids, timestamp, use_reid)

        if use_hungarian and _SCIPY_AVAILABLE:
            return self._hungarian_match(score_matrix, track_ids, threshold)
        else:
            if use_hungarian and not _SCIPY_AVAILABLE:
                logger.debug("scipy bulunamadı — greedy fallback aktif")
            return self._greedy_match(score_matrix, track_ids, threshold)

    def update(self, frame: np.ndarray, raw_detections: List[Any], timestamp: float) -> List[Tuple[Any, int]]:
        """Processes YOLO raw detections and updates BoTSORT tracks."""
        h, w = frame.shape[:2]
        
        # Apply Camera Motion Compensation
        dx, dy = self._compute_cmc(frame)
        for tracker in self.trackers.values():
            tracker.apply_cmc(dx, dy)

        high_dets: List[Dict[str, Any]] = []
        low_dets: List[Dict[str, Any]] = []

        # Parse detections and allocate to high/low groups
        for idx, det in enumerate(raw_detections):
            x1, y1, x2, y2 = det.bbox
            conf = det.confidence
            bev_xy = self._bbox_foot_to_bev(x1, y1, x2, y2)
            crop = frame[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
            
            det_data = {
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "bev": bev_xy,
                "crop": crop,
                "raw_idx": idx
            }
            if conf >= self.track_high_thresh:
                high_dets.append(det_data)
            elif conf >= self.track_low_thresh:
                low_dets.append(det_data)

        active_track_ids = list(self.trackers.keys())
        
        # First Association: Match active tracks with high-confidence detections
        matched_1, unmatched_dets_1, unmatched_tracks_1 = self._associate(
            high_dets, active_track_ids, threshold=0.30, timestamp=timestamp, use_reid=True
        )

        # Second Association: Match remaining tracks with low-confidence detections
        matched_2, unmatched_dets_2, unmatched_tracks_2 = self._associate(
            low_dets, unmatched_tracks_1, threshold=0.15, timestamp=timestamp, use_reid=False
        )

        tracked_results: List[Tuple[Any, int]] = []

        # Process first stage matched tracks
        for tid, det_idx in matched_1.items():
            det = high_dets[det_idx]
            self.reid.register(tid, det["crop"])
            self.reid.verify_id(tid, det["crop"], threshold=0.55)
            self.trackers[tid].update(det["bev"], det["conf"], timestamp)
            self.last_bboxes[tid] = list(det["bbox"])
            tracked_results.append((raw_detections[det["raw_idx"]], tid))

        # Process second stage matched tracks
        for tid, det_idx in matched_2.items():
            det = low_dets[det_idx]
            self.trackers[tid].update(det["bev"], det["conf"], timestamp)
            self.last_bboxes[tid] = list(det["bbox"])
            tracked_results.append((raw_detections[det["raw_idx"]], tid))

        # Initialize new tracks for unmatched high-confidence detections
        for det_idx in unmatched_dets_1:
            det = high_dets[det_idx]
            if det["conf"] >= self.new_track_thresh:
                tid = self.next_id
                self.next_id += 1
                self.trackers[tid] = AdaptiveKalmanTracker(
                    track_id=tid,
                    init_bev=det["bev"],
                    camera_mode=self.camera_mode,
                    init_timestamp=timestamp
                )
                self.last_bboxes[tid] = list(det["bbox"])
                self.reid.register(tid, det["crop"])
                tracked_results.append((raw_detections[det["raw_idx"]], tid))

        # Clean up stale/lost tracks
        observed_ids = {track_id for _, track_id in tracked_results}
        for tid in list(self.trackers.keys()):
            if tid not in observed_ids:
                self.trackers[tid].mark_lost()
                if self.trackers[tid].lost_count > self.track_buffer:
                    self.trackers.pop(tid, None)
                    self.last_bboxes.pop(tid, None)

        self.reid.cleanup(set(self.trackers.keys()))
        return tracked_results


# ══════════════════════════════════════════════════════════════════
# FALLBACK: LightweightTracker (IoU-only, CPU fallback)
# ══════════════════════════════════════════════════════════════════

class LightweightTracker:
    """Simple 2D IoU tracker acting as fallback when Kalman/ReID is disabled or GPU is missing."""

    def __init__(self, match_thresh: float = 0.25) -> None:
        self.match_thresh = match_thresh
        self.active: dict[int, tuple[tuple[int, int, int, int], float]] = {}
        self.next_id = 1

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def assign(self, bbox: tuple[int, int, int, int], timestamp: float) -> int:
        self.active = {key: value for key, value in self.active.items() if timestamp - value[1] <= 2.0}
        matches = [(track_id, self._iou(bbox, state[0])) for track_id, state in self.active.items()]
        if matches:
            track_id, overlap = max(matches, key=lambda item: item[1])
            if overlap >= self.match_thresh:
                self.active[track_id] = (bbox, timestamp)
                return track_id
        track_id = self.next_id
        self.next_id += 1
        self.active[track_id] = (bbox, timestamp)
        return track_id
