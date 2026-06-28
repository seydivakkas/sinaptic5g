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

import re
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

logger = logging.getLogger("sinaptic5g.plate_ocr")

TURKISH_PLATE_PATTERN = re.compile(r'^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}[0-9]{2,4}$')

# Vocabulary for CTC decoding
# Blank character is at index 0
VOCABULARY = "-" + "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


class PlateRecognizer:
    """Two-stage Turkish license plate recognition: LPRNet (bbox) -> CRNN+CTC (character).
    
    Faz 4 iyileştirmeleri:
    - Confidence-weighted temporal voting
    - Edit distance tabanlı gruplama (OCR hatalarına karşı dayanıklı)
    - Stale buffer temizleme
    - Max buffer boyutu ve TTL kontrolü
    """

    LATENCY_MS = 250.8  # Default baseline, updated by benchmark
    STATUS = "ÖLÇÜLDÜ"

    def __init__(
        self,
        lprnet_model_path: str = "models/lprnet.onnx",
        crnn_model_path: str = "models/crnn.onnx",
        voting_buffer_size: int = 7,          # Faz 4: artırıldı (5→7)
        edit_distance_threshold: int = 2,     # Faz 4: edit distance gruplama
        min_votes_for_confidence: int = 3,    # Faz 4: minimum oy sayısı
        max_buffer_age_frames: int = 60,      # Faz 4: stale buffer temizleme
    ):
        self.voting_buffer_size = voting_buffer_size
        self.edit_distance_threshold = edit_distance_threshold
        self.min_votes_for_confidence = min_votes_for_confidence
        self.max_buffer_age_frames = max_buffer_age_frames

        # Faz 4: (plate_text, confidence) tuple buffer
        self._plate_buffer: List[Tuple[str, float]] = []
        self._buffer_frame_counter: int = 0
        self._last_access_frame: int = 0

        self.lprnet_session = None
        self.crnn_session = None

        # Load models using ONNX Runtime
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]

            # Try loading LPRNet
            lpr_path = Path(lprnet_model_path)
            if not lpr_path.is_file():
                lpr_path = Path(__file__).resolve().parent.parent / lprnet_model_path
            if lpr_path.is_file():
                lpr_opts = ort.SessionOptions()
                lpr_opts.intra_op_num_threads = 1
                lpr_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                lpr_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                self.lprnet_session = ort.InferenceSession(str(lpr_path), lpr_opts, providers=providers)
                logger.info("Loaded LPRNet ONNX model successfully.")
            else:
                logger.warning("LPRNet ONNX model file not found: %s", lprnet_model_path)

            # Try loading CRNN
            crnn_path = Path(crnn_model_path)
            if not crnn_path.is_file():
                crnn_path = Path(__file__).resolve().parent.parent / crnn_model_path
            if crnn_path.is_file():
                crnn_opts = ort.SessionOptions()
                crnn_opts.intra_op_num_threads = 2
                crnn_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                crnn_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                self.crnn_session = ort.InferenceSession(str(crnn_path), crnn_opts, providers=providers)
                logger.info("Loaded CRNN ONNX model successfully.")
            else:
                logger.warning("CRNN ONNX model file not found: %s", crnn_model_path)

        except Exception as e:
            logger.error("Failed to initialize ONNX Runtime for LPRNet/CRNN: %s", e)

    def recognize(self, frame: np.ndarray) -> Optional[Tuple[str, float]]:
        """Detects plate region, runs OCR, and validates the text format.
        Returns Tuple[plate_text, confidence] or None.
        """
        self._buffer_frame_counter += 1
        self._last_access_frame = self._buffer_frame_counter

        if frame is None or frame.size == 0:
            return self._get_voted_plate()

        # Step 1: Detect plate region
        plate_region = self._detect_plate_region(frame)
        if plate_region is None or plate_region.size == 0:
            return self._get_voted_plate()

        # Step 2: Preprocess
        preprocessed = self._preprocess(plate_region)

        # Step 3: Run OCR
        res = self._run_ocr(preprocessed)
        if res is None:
            return self._get_voted_plate()

        plate_text, conf = res

        # Step 4: Validate and format
        valid_text = self._validate_and_format(plate_text)
        if valid_text:
            self._add_to_buffer(valid_text, conf)
            return valid_text, conf

        return self._get_voted_plate()

    def _detect_plate_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detects license plate bounding box using LPRNet if available."""
        if self.lprnet_session is None:
            # Fallback: assume the frame passed in is already a close crop of the plate
            return frame

        try:
            h_orig, w_orig = frame.shape[:2]
            # Resize image to LPRNet input size (typically 320x96 or similar)
            input_w, input_h = 320, 96
            resized = cv2.resize(frame, (input_w, input_h))
            blob = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            blob = np.expand_dims(blob, axis=0)

            input_name = self.lprnet_session.get_inputs()[0].name
            outputs = self.lprnet_session.run(None, {input_name: blob})
            
            # Extract plate bounding box from output tensor
            # Assume LPRNet outputs bounding box coordinates in shape [1, 4]: [x1, y1, x2, y2]
            coords = outputs[0][0]
            x1 = int(coords[0] * w_orig)
            y1 = int(coords[1] * h_orig)
            x2 = int(coords[2] * w_orig)
            y2 = int(coords[3] * h_orig)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_orig, x2)
            y2 = min(h_orig, y2)
            
            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2]
        except Exception as e:
            logger.debug("LPRNet bbox detection failed: %s", e)
        
        return frame

    def _preprocess(self, plate_img: np.ndarray) -> np.ndarray:
        """Applies CLAHE preprocessing and resizes to CRNN expected size."""
        h, w = plate_img.shape[:2]
        # Resize to standard CRNN input size (160x32)
        target_w, target_h = 160, 32
        resized = cv2.resize(plate_img, (target_w, target_h))

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # CLAHE (clipLimit=2.0, tileGridSize=8x8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def _run_ocr(self, plate_img: np.ndarray) -> Optional[Tuple[str, float]]:
        """Runs CRNN character recognition with CTC greedy decoding."""
        if self.crnn_session is None:
            return None

        try:
            # Normalize to [-1, 1]
            blob = (plate_img.astype(np.float32) / 127.5) - 1.0
            blob = np.expand_dims(np.expand_dims(blob, axis=0), axis=0) # Shape: [1, 1, 32, 160]

            input_name = self.crnn_session.get_inputs()[0].name
            outputs = self.crnn_session.run(None, {input_name: blob})
            
            # CRNN output shape: [seq_len, batch_size, num_classes] or [batch_size, seq_len, num_classes]
            output_tensor = outputs[0]
            if output_tensor.ndim == 3:
                # Transpose to [seq_len, num_classes] (assuming batch size = 1)
                if output_tensor.shape[0] == 1:
                    preds = output_tensor[0]
                else:
                    preds = output_tensor[:, 0, :]
            else:
                preds = output_tensor

            # CTC Greedy Decoding
            # Get argmax for each time step
            char_indices = np.argmax(preds, axis=-1)
            probs = np.max(preds, axis=-1)
            
            # Apply softmax to probs if values are raw logits
            # For simplicity, we can calculate a pseudo confidence score by averaging probs
            mean_conf = float(np.mean(probs))

            decoded_chars = []
            prev_idx = -1
            for idx in char_indices:
                if idx != 0 and idx != prev_idx:  # Ignore blank (0) and duplicates
                    if idx < len(VOCABULARY):
                        decoded_chars.append(VOCABULARY[idx])
                prev_idx = idx

            plate_text = "".join(decoded_chars)
            return plate_text, mean_conf
        except Exception as e:
            logger.error("CRNN OCR extraction failed: %s", e)
            return None

    def _validate_and_format(self, raw_text: str) -> Optional[str]:
        """Cleans and validates the license plate using Turkish formatting rules."""
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        if TURKISH_PLATE_PATTERN.match(cleaned):
            return cleaned
        return None

    def _add_to_buffer(self, plate_text: str, confidence: float) -> None:
        """Faz 4: Confidence ile birlikte buffer'a ekler. Stale kontrol yapar."""
        # Stale buffer temizleme: max_buffer_age_frames karenin üzerindeyse sıfırla
        if (self._buffer_frame_counter - self._last_access_frame) > self.max_buffer_age_frames:
            self._plate_buffer.clear()
            logger.debug("Stale plate buffer temizlendi.")

        self._plate_buffer.append((plate_text, confidence))
        if len(self._plate_buffer) > self.voting_buffer_size:
            self._plate_buffer.pop(0)

    def _get_voted_plate(self) -> Optional[Tuple[str, float]]:
        """
        Faz 4: Confidence-weighted temporal voting + edit distance gruplama.
        
        Strateji:
        1. Edit distance <= threshold olanları aynı kümeye al
        2. Her küme için toplam confidence skoru hesapla
        3. En yüksek skorlu kümenin temsilcisini döndür
        4. Minimum oy koşulunu kontrol et
        """
        if not self._plate_buffer:
            return None

        if len(self._plate_buffer) < self.min_votes_for_confidence:
            # Yeterli oy yok — en güvenilir tek tahmini döndür
            if self._plate_buffer:
                best = max(self._plate_buffer, key=lambda x: x[1])
                return best[0], best[1] * 0.7  # Düşük güven çarpanı
            return None

        # Edit distance gruplama
        groups: List[List[Tuple[str, float]]] = []

        for plate_text, conf in self._plate_buffer:
            placed = False
            for group in groups:
                representative = group[0][0]
                if _edit_distance(plate_text, representative) <= self.edit_distance_threshold:
                    group.append((plate_text, conf))
                    placed = True
                    break
            if not placed:
                groups.append([(plate_text, conf)])

        # Her grup için toplam confidence hesapla
        best_group = None
        best_score = -1.0

        for group in groups:
            total_conf = sum(conf for _, conf in group)
            vote_bonus = len(group) * 0.1  # Her ekstra oy için bonus
            score = total_conf + vote_bonus
            if score > best_score:
                best_score = score
                best_group = group

        if best_group is None or len(best_group) < self.min_votes_for_confidence:
            return None

        # Temsilci: en yüksek confidence'lı eleman
        best_text, best_conf = max(best_group, key=lambda x: x[1])
        # Confidence normalize et: oy sayısına göre artır
        normalized_conf = min(1.0, best_conf * (1.0 + 0.05 * len(best_group)))

        return best_text, normalized_conf

    def clear_buffer(self) -> None:
        """Buffer'ı temizler ve sayaçları sıfırlar."""
        self._plate_buffer.clear()
        self._buffer_frame_counter = 0
        self._last_access_frame = 0
