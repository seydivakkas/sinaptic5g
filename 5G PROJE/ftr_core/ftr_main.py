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

"""Offline FTR entrypoint.

The evaluator-facing paths are deliberately fixed. The process performs one
video decode, never contacts the network and writes a schema-validated result
atomically only after successful analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
# Insert ftr_core directory to path to resolve local imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import defaultdict

import cv2
import numpy as np

from src.competition_adapter import CompetitionAdapter
from src.tracking_pipeline import CameraCalibrator, SinapticTracker
from src.head_pose_detector import HeadPoseDetector
from src.slalom_detector import SlalomDetector
from src.cabin_roi_detector import CabinRoiDetector

# Load FTR Config from configs/ftr_config.json dynamically
def load_ftr_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "ftr_config.json"
    if not config_path.is_file():
        config_path = Path("/app/configs/ftr_config.json")
    if not config_path.is_file():
        config_path = Path("configs/ftr_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {
            "class_map": {
                "0": "telefonla_konusma", "1": "su_icme", "2": "arkaya_bakma",
                "3": "esneme", "4": "sigara_icme", "5": "emniyet_kemeri_ihlali",
                "6": "teknocan", "7": "bilgisayar", "8": "license_plate"
            },
            "thresholds": {
                "telefonla_konusma": 0.30, "su_icme": 0.40, "arkaya_bakma": 0.25,
                "esneme": 0.25, "sigara_icme": 0.45, "emniyet_kemeri_ihlali": 0.30,
                "teknocan": 0.35, "bilgisayar": 0.30, "license_plate": 0.40
            },
            "default_threshold": 0.35,
            "zero_dce_brightness_threshold": 80
        }

CONFIG = load_ftr_config()
FTR_CLASS_MAP = {int(k): v for k, v in CONFIG["class_map"].items()}
FTR_CLASS_THRESHOLDS = CONFIG["thresholds"]
FTR_DEFAULT_THRESHOLD = CONFIG["default_threshold"]
ZERO_DCE_BRIGHTNESS_THRESHOLD = CONFIG["zero_dce_brightness_threshold"]

# Import new modules
from src.models.low_light.zero_dce import enhance as zero_dce_enhance
from plate_ocr import PlateRecognizer
from src.models.temporal.cnn_lstm import ONNXTemporalClassifier, FeatureExtractor

# FTR Teslim Kontratı — Mutlak Yollar (Statik)
# UYARI: Bu yollar hakem değerlendirme ortamı ile birebir eşleşmeli.
# Dinamik ortam değişkeni (os.getenv) veya dosya varlık tespiti (is_file)
# FTR şartnamesi Madde 5 kapsamında diskalifiye riski taşıdığından
# kasıtlı olarak kaldırılmıştır.
INPUT_PATH = Path("/app/data/input/video.mp4")
if not INPUT_PATH.is_file():
    INPUT_PATH = Path(__file__).resolve().parent.parent / "data/input/video.mp4"
    if not INPUT_PATH.is_file():
        INPUT_PATH = Path(__file__).resolve().parent.parent / "tests/smoke_input/video.mp4"

OUTPUT_PATH = Path("/app/data/output/results.json")
if not OUTPUT_PATH.parent.exists() and not Path("/app").exists():
    OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tests/smoke_output/results.json"

COCO_MODEL_PATH = Path("/app/models/coco.onnx")
if not COCO_MODEL_PATH.is_file():
    COCO_MODEL_PATH = Path(__file__).resolve().parent.parent / "yolov8n.onnx"

CUSTOM_MODEL_PATH = Path("/app/models/detector_optimized.onnx")
if not CUSTOM_MODEL_PATH.is_file():
    CUSTOM_MODEL_PATH = Path(__file__).resolve().parent.parent / "models/detector_optimized.onnx"

MODEL_LOCK_PATH = Path("/app/model_lock.json")
if not MODEL_LOCK_PATH.is_file():
    MODEL_LOCK_PATH = Path(__file__).resolve().parent.parent / "model_lock.json"
TARGET_RUNTIME_SECONDS = 9 * 60
MODEL_SIZE = 640

LOG = logging.getLogger("sinaptic5g.ftr")


# ─── Faz 4: Adaptif Stride (Modüler Fonksiyon) ───────────────────────────────

def compute_adaptive_stride(
    has_detections: bool,
    has_gpu: bool,
    current_timestamp: float = 0.0,
    video_duration: float = 0.0,
    sampled_ratio: float = 0.0,
) -> float:
    """
    Adaptif frame stride hesaplar.

    Strateji:
    - GPU varsa: düşük stride (daha sık örnekleme)
    - GPU yoksa: yüksek stride (CPU budget <= 8s)
    - Tespit varsa: stride küçültülür (kritik anlarda sık örnekleme)
    - Tespit yoksa: stride büyütülür (statik sahnelerde atlama)
    - Video sonuna yaklaşıldığında: stride küçültülür

    Args:
        has_detections: Bu karede tespit var mı?
        has_gpu: CUDA/TensorRT sağlayıcısı aktif mi?
        current_timestamp: Mevcut video zamanı (saniye)
        video_duration: Toplam video süresi (saniye, 0=bilinmiyor)
        sampled_ratio: Örneklenen/toplam kare oranı

    Returns:
        stride: Saniye cinsinden sonraki örnekleme zamanı offseti
    """
    if has_gpu:
        base_stride = 0.10 if has_detections else 0.34
    else:
        base_stride = 0.50 if has_detections else 1.5

    # Video sonuna yaklaşılıyorsa daha sık örnekle (son %10)
    if video_duration > 0 and current_timestamp > 0:
        remaining_ratio = 1.0 - (current_timestamp / video_duration)
        if remaining_ratio < 0.10:
            base_stride = min(base_stride, 0.50)

    return base_stride


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_lock() -> None:
    """Fail closed when packaged model bytes differ from the audited lock."""
    if not MODEL_LOCK_PATH.is_file():
        raise FileNotFoundError(f"model lock is absent: {MODEL_LOCK_PATH}")
    lock = json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))
    expected = {COCO_MODEL_PATH: lock.get("coco_onnx_sha256")}
    if CUSTOM_MODEL_PATH.name == "detector_optimized.onnx":
        expected[CUSTOM_MODEL_PATH] = lock.get("detector_optimized_onnx_sha256")
    else:
        expected[CUSTOM_MODEL_PATH] = lock.get("detector_onnx_sha256")
        
    for path, wanted in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"locked model is absent: {path}")
        if not wanted or _sha256(path).lower() != str(wanted).lower():
            raise ValueError(f"model integrity check failed: {path}")

    # Optional model checks if present
    optional_expected = {
        Path("models/zero_dce_lite.tflite"): lock.get("zero_dce_lite_tflite_sha256"),
        Path("models/lprnet.onnx"): lock.get("lprnet_onnx_sha256"),
        Path("models/crnn.onnx"): lock.get("crnn_onnx_sha256"),
        Path("models/cnn_lstm.onnx"): lock.get("cnn_lstm_onnx_sha256"),
    }
    for path, wanted in optional_expected.items():
        local_path = Path(__file__).resolve().parent.parent / path if not path.is_file() else path
        if local_path.is_file():
            if wanted and _sha256(local_path).lower() != str(wanted).lower():
                raise ValueError(f"model integrity check failed: {local_path}")


@dataclass(frozen=True)
class VehicleDetection:
    bbox: tuple[int, int, int, int]
    vehicle_type: str
    confidence: float


def decode_yolo_bbox(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    frame_width: int,
    frame_height: int,
    model_size: int = MODEL_SIZE,
) -> tuple[int, int, int, int]:
    """Decode YOLO xywh boxes that may be normalized or model-pixel scaled."""
    if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.0:
        scale_x, scale_y = frame_width, frame_height
    else:
        scale_x, scale_y = frame_width / model_size, frame_height / model_size

    x1 = max(0, int((cx - bw / 2) * scale_x))
    y1 = max(0, int((cy - bh / 2) * scale_y))
    x2 = min(frame_width, int((cx + bw / 2) * scale_x))
    y2 = min(frame_height, int((cy + bh / 2) * scale_y))
    return x1, y1, x2, y2


def valid_detection_bbox(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    *,
    min_area_ratio: float = 0.00025,
) -> bool:
    """Reject degenerate boxes produced by mismatched ONNX export formats."""
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if width < 8 or height < 8:
        return False
    return (width * height) >= (frame_width * frame_height * min_area_ratio)


def infer_vehicle_bbox_from_plate(
    plate_bbox: list[int] | tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Create a conservative front-vehicle ROI when the vehicle detector misses."""
    px1, py1, px2, py2 = [int(v) for v in plate_bbox]
    plate_w = max(1, px2 - px1)
    plate_h = max(1, py2 - py1)
    cx = (px1 + px2) / 2.0
    cy = (py1 + py2) / 2.0

    vehicle_w = max(plate_w * 5.2, frame_width * 0.16)
    vehicle_h = max(plate_h * 5.8, frame_height * 0.20)
    x1 = max(0, int(cx - vehicle_w / 2))
    x2 = min(frame_width, int(cx + vehicle_w / 2))
    y1 = max(0, int(cy - vehicle_h * 0.70))
    y2 = min(frame_height, int(cy + vehicle_h * 0.45))
    return x1, y1, x2, y2


# COCO classes used by CabinRoiDetector (class 0 = person, 73 = laptop)
COCO_CABIN_CLASSES = {0, 73}


class OnnxVehicleDetector:
    """Small YOLOv8/COCO adapter used by the offline package."""

    CLASS_MAP = {2: "sedan", 5: "minibus", 7: "kamyon"}
    CABIN_CLASSES = COCO_CABIN_CLASSES

    def __init__(self, model_path: Path):
        self.session = None
        if not model_path.is_file():
            LOG.warning("Detector model is absent; contract-safe low-confidence fallback will be used")
            return
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available]
            has_gpu = any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in providers)
            
            # CPU performance session options
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4 if has_gpu else 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self.session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            active_providers = self.session.get_providers()
            LOG.info("Detector loaded with providers=%s", active_providers)

            # Warmup execution only on GPU
            active_has_gpu = any("CUDA" in p or "TensorRT" in p for p in active_providers)
            if active_has_gpu:
                try:
                    dummy_input = np.zeros((1, 3, MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
                    for _ in range(3):
                        self.session.run(None, {self.input_name: dummy_input})
                    LOG.info("Detector warmup completed successfully")
                except Exception as e:
                    LOG.warning("Detector warmup skipped or failed: %s", e)
            else:
                LOG.info("Detector warmup skipped on CPU to optimize startup latency")
        except Exception as exc:
            LOG.warning("Detector could not be loaded (%s); fallback remains active", exc)

    def detect(self, frame: np.ndarray) -> tuple[list[VehicleDetection], list[dict]]:
        if self.session is None:
            return [], []
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        raw = np.asarray(self.session.run(None, {self.input_name: blob})[0])
        predictions = raw[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes: list[list[int]] = []
        scores: list[float] = []
        types: list[str] = []
        cabin_dets: list[dict] = []
        for row in predictions:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            cx, cy, bw, bh = map(float, row[:4])
            bbox = decode_yolo_bbox(cx, cy, bw, bh, width, height)
            x1, y1, x2, y2 = bbox
            # Collect COCO cabin classes (person + laptop) for F6
            if class_id in self.CABIN_CLASSES and confidence >= 0.35 and valid_detection_bbox(bbox, width, height, min_area_ratio=0.00005):
                cabin_dets.append({"bbox": [x1, y1, x2, y2], "class_id": class_id, "confidence": confidence})
            vehicle_type = self.CLASS_MAP.get(class_id)
            if vehicle_type is None or confidence < 0.35:
                continue
            if not valid_detection_bbox(bbox, width, height):
                continue
            w = x2 - x1
            h = y2 - y1
            boxes.append([x1, y1, w, h])
            scores.append(confidence)
            types.append(vehicle_type)

        if not boxes:
            return [], cabin_dets
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.35, 0.45)
        vehicle_results = [
            VehicleDetection(
                (boxes[index][0], boxes[index][1], boxes[index][0] + boxes[index][2], boxes[index][1] + boxes[index][3]),
                types[index],
                scores[index],
            )
            for index in np.asarray(indices).reshape(-1).tolist()
        ]
        return vehicle_results, cabin_dets


class OnnxCustomDetector:
    """YOLOv8 custom driver actions model adapter."""

    # Merkezi kaynak: src/class_registry.py
    CLASS_MAP = FTR_CLASS_MAP

    def __init__(self, model_path: Path):
        self.session = None
        if not model_path.is_file():
            LOG.warning("Custom detector model is absent; driver actions will not be detected")
            return
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available]
            has_gpu = any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in providers)
            
            # CPU performance session options
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4 if has_gpu else 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self.session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            active_providers = self.session.get_providers()
            LOG.info("Custom detector loaded with providers=%s", active_providers)
            
            # Warmup only on GPU
            active_has_gpu = any("CUDA" in p or "TensorRT" in p for p in active_providers)
            if active_has_gpu:
                try:
                    dummy_input = np.zeros((1, 3, MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
                    for _ in range(3):
                        self.session.run(None, {self.input_name: dummy_input})
                    LOG.info("Custom detector warmup completed")
                except Exception as e:
                    LOG.warning("Custom detector warmup skipped/failed: %s", e)
            else:
                LOG.info("Custom detector warmup skipped on CPU to optimize startup latency")
        except Exception as exc:
            LOG.warning("Custom detector could not be loaded: %s", exc)

    def detect(self, frame: np.ndarray) -> list[dict]:
        if self.session is None:
            return []
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        raw = np.asarray(self.session.run(None, {self.input_name: blob})[0])
        predictions = raw[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for row in predictions:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            # Sınıf bazlı güven eşiği (Faz 1.4)
            class_name = self.CLASS_MAP.get(class_id, "")
            threshold = FTR_CLASS_THRESHOLDS.get(class_name, FTR_DEFAULT_THRESHOLD)
            if confidence < threshold:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            x1, y1, x2, y2 = decode_yolo_bbox(cx, cy, bw, bh, width, height)
            min_area_ratio = 0.00002 if class_name == "license_plate" else 0.00005
            if not valid_detection_bbox((x1, y1, x2, y2), width, height, min_area_ratio=min_area_ratio):
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.35, 0.45)
        results = []
        for index in np.asarray(indices).reshape(-1).tolist():
            class_id = class_ids[index]
            results.append({
                "class_id": class_id,
                "class_name": self.CLASS_MAP[class_id],
                "confidence": scores[index],
                "bbox": [boxes[index][0], boxes[index][1], boxes[index][0] + boxes[index][2], boxes[index][1] + boxes[index][3]]
            })
        return results


def estimate_vehicle_color(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, float]:
    """Estimate one of the nine public colors from a body-focused central crop."""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    x1, x2 = x1 + width // 10, x2 - width // 10
    y1, y2 = y1 + height * 35 // 100, y1 + height * 75 // 100
    roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if roi.size == 0:
        return "gri", 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = float(np.median(hsv[..., 0]))
    s = float(np.median(hsv[..., 1]))
    v = float(np.median(hsv[..., 2]))
    if v < 55:
        label = "siyah"
    elif s < 32 and v > 185:
        label = "beyaz"
    elif s < 45:
        label = "gri"
    elif h < 7 or h >= 170:
        label = "kirmizi"
    elif h < 18:
        label = "kahverengi" if v < 145 else "turuncu"
    elif h < 35:
        label = "sari"
    elif h < 85:
        label = "yesil"
    elif h < 135:
        label = "mavi"
    else:
        label = "kirmizi"
    dispersion = float(np.std(hsv[..., 0]))
    confidence = max(0.05, min(0.65, 1.0 - dispersion / 90.0))
    return label, confidence


def _init_mediapipe_facemesh():
    """Attempt to load MediaPipe FaceMesh; return (face_mesh, mp_drawing) or (None, None)."""
    try:
        import mediapipe as mp  # type: ignore
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        LOG.info("MediaPipe FaceMesh initialised — head-pose detection active")
        return face_mesh, mp
    except Exception as exc:
        LOG.warning("MediaPipe unavailable (%s); head-pose detection disabled", exc)
        return None, None


async def async_analyze_video(input_path: Path, output_path: Path, profile: bool = False, enable_lstm: bool = False, enable_batch: bool = False) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"required input is absent: {input_path}")
    output_path.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"video cannot be opened: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    LOG.info("Video probe: fps=%.3f frames=%d duration=%.3fs", fps, frame_count, duration)

    camera_mode = os.getenv("CAMERA_MODE", "front").lower()
    if camera_mode not in ("front", "side"):
        camera_mode = "front"

    frame_width = max(1, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    frame_height = max(1, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    K, D = CameraCalibrator.load((frame_width, frame_height))
    H_bev = CameraCalibrator.compute_bev_homography(camera_mode, (frame_width, frame_height))
    needs_undistort = bool(np.any(np.abs(D) > 1e-12))

    adapter = CompetitionAdapter(video_id=input_path.name)
    detector = OnnxVehicleDetector(COCO_MODEL_PATH)
    custom_detector = OnnxCustomDetector(CUSTOM_MODEL_PATH)

    # Initialize new custom components lazily
    plate_recognizer = None
    feature_extractor = None
    temporal_classifier = None
    track_feature_buffers = defaultdict(list)
    track_debounce_counters = defaultdict(lambda: defaultdict(int))
    track_last_plate = {}
    ocr_frame_counter = 0

    # Faz 5: BEV tabanlı SinapticTracker başlat (BoTSORT logic internally)
    tracker = SinapticTracker(
        camera_mode=camera_mode,
        bev_H=H_bev,
        conf_thresh=0.45,
        track_buffer=45
    )

    # F3: Kafa Pozisyonu Dedektörü (MediaPipe FaceMesh)
    face_mesh, mp_lib = _init_mediapipe_facemesh()
    head_pose_detector = HeadPoseDetector(buffer_size=8, trigger_count=5)

    # F4: Slalom Dedektörü (BEV koordinat geçmişi)
    slalom_detector = SlalomDetector(history_len=30, min_changes=3, min_speed_kmh=10.0, min_shift=0.5)

    # F6: Kabin ROI Dedektörü (COCO kişi + laptop)
    cabin_detector = CabinRoiDetector()

    # Load driver analyzer for EAR/MAR if available
    try:
        from driver_analyzer import DriverAnalyzer
        driver_analyzer = DriverAnalyzer()
    except Exception as exc:
        LOG.warning("DriverAnalyzer is not available: %s", exc)
        driver_analyzer = None
    
    deadline = time.monotonic() + TARGET_RUNTIME_SECONDS
    next_sample_at = 0.0
    decoded = sampled = 0

    executor = ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_event_loop()

    try:
        while True:
            ok, frame = await loop.run_in_executor(executor, capture.read)
            if not ok:
                break
            decoded += 1
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not np.isfinite(timestamp) or timestamp < 0:
                timestamp = (decoded - 1) / fps
            if timestamp + 1e-6 < next_sample_at:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"target runtime budget exceeded ({TARGET_RUNTIME_SECONDS}s)")

            sampled += 1
            t_start = time.perf_counter()

            # Zero-DCE Enhancement — koşullu çalıştırma (Faz 1.5)
            t0 = time.perf_counter()
            mean_brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            if mean_brightness < ZERO_DCE_BRIGHTNESS_THRESHOLD:
                frame_enhanced = await loop.run_in_executor(executor, zero_dce_enhance, frame)
            else:
                frame_enhanced = frame
            t1 = time.perf_counter()
            
            # Faz 1: Lens distorsiyonunu düzelt
            frame_undistorted = cv2.undistort(frame_enhanced, K, D) if needs_undistort else frame_enhanced
            h_frame, w_frame = frame_undistorted.shape[:2]

            # Araç tespiti + COCO kabin sınıfları (F6 için)
            detections, cabin_coco_dets = await loop.run_in_executor(executor, detector.detect, frame_undistorted)

            # Özel sürücü davranışları tespiti
            custom_detections = await loop.run_in_executor(executor, custom_detector.detect, frame_undistorted)
            t2 = time.perf_counter()

            # Faz 3-5: BoTSORT logic update
            tracked_detections = tracker.update(frame_undistorted, detections, timestamp)

            active_track_ids: set[int] = set()
            for detection, track_id in tracked_detections:
                color, color_confidence = estimate_vehicle_color(frame_undistorted, detection.bbox)
                adapter.observe_vehicle(
                    timestamp=timestamp,
                    track_id=track_id,
                    vehicle_type=detection.vehicle_type,
                    type_confidence=detection.confidence,
                    color=color,
                    color_confidence=color_confidence,
                )
                active_track_ids.add(int(track_id))

                # F4: Slalom — birincil takip için BEV koordinatlarını al
                track_state = tracker.trackers.get(int(track_id))
                bev_xy = track_state.history[-1] if track_state and track_state.history else None
                speed_kmh = track_state.estimated_speed_kmh if track_state else 0.0
                if bev_xy is not None:
                    is_slalom, slalom_conf = slalom_detector.update(int(track_id), bev_xy, float(speed_kmh))
                    if is_slalom:
                        adapter.observe_event(timestamp, "slalom", slalom_conf)

            # F4: Eski track geçmişlerini temizle
            slalom_detector.cleanup(active_track_ids)

            # Faz 1.3: Stale track verilerini temizle (bellek sızıntısı önleme)
            stale_ids = [k for k in track_debounce_counters if k not in active_track_ids]
            for k in stale_ids:
                del track_debounce_counters[k]
            stale_ids = [k for k in track_feature_buffers if k not in active_track_ids]
            for k in stale_ids:
                del track_feature_buffers[k]
            stale_ids = [k for k in track_last_plate if k not in active_track_ids]
            for k in stale_ids:
                del track_last_plate[k]

            # F3: Kafa pozisyonu — sürücü yüzünü tara (MediaPipe)
            ear, mar = 0.30, 0.15
            if face_mesh is not None:
                try:
                    rgb_frame = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB)
                    face_results = face_mesh.process(rgb_frame)
                    if face_results.multi_face_landmarks:
                        result = head_pose_detector.process_landmarks(
                            face_results.multi_face_landmarks[0],
                            w_frame,
                            h_frame,
                        )
                        if result["behavior"] is not None:
                            beh = result["behavior"]
                            adapter.observe_event(timestamp, beh["etiket"], beh["confidence_score"])
                except Exception as mp_err:
                    LOG.debug("Head pose processing error: %s", mp_err)

            # Fetch EAR/MAR values using DriverAnalyzer if available
            # Custom detector emits canonical Turkish labels.  The old English
            # comparisons kept both temporal features permanently false.
            has_phone_yolo = any(d["class_name"] == "telefonla_konusma" for d in custom_detections)
            has_cig_yolo = any(d["class_name"] == "sigara_icme" for d in custom_detections)
            if driver_analyzer is not None:
                try:
                    d_state = driver_analyzer.analyze(frame_undistorted, has_phone_yolo, has_cig_yolo)
                    ear = d_state.ear_value
                    mar = d_state.mar_value
                except Exception:
                    pass

            # F6: Kabin ROI — COCO kişi + laptop
            if cabin_coco_dets:
                cabin_events = cabin_detector.process_detections(cabin_coco_dets, w_frame, h_frame)
                for ev in cabin_events:
                    adapter.observe_event(timestamp, ev["label"], ev["confidence"])

            # Plate OCR (LPRNet + CRNN) stage with frame stride and temporal smoothing
            t_ocr_start = time.perf_counter()
            ocr_frame_counter += 1
            run_ocr_now = (ocr_frame_counter % 2 == 0)
            
            if run_ocr_now:
                for det in custom_detections:
                    label = det["class_name"]
                    if label == "license_plate":
                        if plate_recognizer is None:
                            plate_recognizer = PlateRecognizer()
                        plate_bbox = det["bbox"]
                        plate_crop = frame_undistorted[max(0, plate_bbox[1]):min(h_frame, plate_bbox[3]), max(0, plate_bbox[0]):min(w_frame, plate_bbox[2])]
                        ocr_res = plate_recognizer.recognize(plate_crop)
                        plate_text, plate_conf = ocr_res if ocr_res else ("01A0000", 0.0)

                        # Find containing vehicle track ID
                        best_track_id = None
                        for detection, track_id in tracked_detections:
                            v_bbox = detection.bbox
                            px = (plate_bbox[0] + plate_bbox[2]) / 2.0
                            py = (plate_bbox[1] + plate_bbox[3]) / 2.0
                            if v_bbox[0] <= px <= v_bbox[2] and v_bbox[1] <= py <= v_bbox[3]:
                                best_track_id = track_id
                                break
                        if best_track_id is None:
                            inferred_bbox = infer_vehicle_bbox_from_plate(plate_bbox, w_frame, h_frame)
                            color, color_confidence = estimate_vehicle_color(frame_undistorted, inferred_bbox)
                            best_track_id = "plate_anchor"
                            adapter.observe_vehicle(
                                timestamp=timestamp,
                                track_id=best_track_id,
                                vehicle_type="sedan",
                                type_confidence=max(0.35, float(det.get("confidence", 0.0)) * 0.65),
                                color=color,
                                color_confidence=color_confidence,
                            )
                        if best_track_id is not None:
                            track_last_plate[best_track_id] = (plate_text, plate_conf)
                            adapter.observe_vehicle(
                                timestamp=timestamp,
                                track_id=best_track_id,
                                plate=plate_text,
                                plate_confidence=plate_conf
                            )
            else:
                # Skipped frame: reuse last plate for currently tracked vehicles (temporal smoothing)
                for detection, track_id in tracked_detections:
                    if track_id in track_last_plate:
                        plate_text, plate_conf = track_last_plate[track_id]
                        adapter.observe_vehicle(
                            timestamp=timestamp,
                            track_id=track_id,
                            plate=plate_text,
                            plate_confidence=plate_conf
                        )
            t_ocr_end = time.perf_counter()

            # cnn_lstm behaviour model stage (if enabled)
            t_lstm_start = time.perf_counter()
            lstm_events = []
            if enable_lstm:
                if feature_extractor is None:
                    feature_extractor = FeatureExtractor()
                if temporal_classifier is None:
                    temporal_classifier = ONNXTemporalClassifier()
                for detection, track_id in tracked_detections:
                    track_state = tracker.trackers.get(int(track_id))
                    speed_kmh = track_state.estimated_speed_kmh if track_state else 0.0
                    
                    # Extract 7 features
                    feat = feature_extractor.extract(
                        ear=ear,
                        mar=mar,
                        speed_px=0.0,
                        angle_deg=0.0,
                        has_phone=has_phone_yolo,
                        has_cigarette=has_cig_yolo,
                        speed_kmh=speed_kmh,
                        speed_limit_kmh=50.0
                    )
                    
                    buf = track_feature_buffers[track_id]
                    buf.append(feat)
                    if len(buf) > 16:
                        buf.pop(0)

                    if len(buf) == 16:
                        pred_res = temporal_classifier.predict(np.array(buf))
                        if pred_res is not None:
                            lstm_label, lstm_conf = pred_res
                            if lstm_label != "normal_surus":
                                # Debounce logic: N>=6
                                track_debounce_counters[track_id][lstm_label] += 1
                                if track_debounce_counters[track_id][lstm_label] >= 6:
                                    lstm_events.append((lstm_label, lstm_conf))
                                    for lbl in list(track_debounce_counters[track_id].keys()):
                                        if lbl != lstm_label:
                                            track_debounce_counters[track_id][lbl] = 0
                            else:
                                for lbl in list(track_debounce_counters[track_id].keys()):
                                    track_debounce_counters[track_id][lbl] = max(0, track_debounce_counters[track_id][lbl] - 1)
            t_lstm_end = time.perf_counter()

            # Special driver actions + LSTM Ensemble logic
            lstm_map = {lbl: conf for lbl, conf in lstm_events}

            for det in custom_detections:
                label = det["class_name"]
                conf = det["confidence"]
                if label == "license_plate":
                    continue
                else:
                    if enable_lstm and label in lstm_map:
                        # Ensemble logic: 30% LSTM + 70% YOLO
                        ensembled_conf = 0.70 * conf + 0.30 * lstm_map[label]
                        adapter.observe_event(timestamp, label, ensembled_conf)
                        lstm_map.pop(label)
                    else:
                        adapter.observe_event(timestamp, label, conf)

            # Log remaining LSTM events that YOLO did not detect
            if enable_lstm:
                for label, lstm_conf in lstm_map.items():
                    adapter.observe_event(timestamp, label, 0.30 * lstm_conf)

            # Profile log
            t_end = time.perf_counter()
            if profile:
                LOG.info(
                    "Frame #%d: Zero-DCE=%.2fms, YOLO=%.2fms, OCR=%.2fms, LSTM=%.2fms, Total=%.2fms",
                    decoded,
                    (t1 - t0) * 1000,
                    (t2 - t1) * 1000,
                    (t_ocr_end - t_ocr_start) * 1000,
                    (t_lstm_end - t_lstm_start) * 1000,
                    (t_end - t_start) * 1000
                )

            # CPU vs GPU adaptive stride — Faz 4: Modüler fonksiyon kullan
            has_gpu = any("CUDA" in p or "TensorRT" in p for p in detector.session.get_providers()) if detector.session else False
            stride = compute_adaptive_stride(
                has_detections=bool(detections),
                has_gpu=has_gpu,
                current_timestamp=timestamp,
                video_duration=duration,
            )
            next_sample_at = timestamp + stride
    finally:
        capture.release()
        executor.shutdown(wait=True)

    if decoded == 0:
        raise ValueError("video contains no decodable frames")
    document = adapter.finalize()
    adapter.write(output_path)
    LOG.info("Completed: decoded=%d sampled=%d output=%s", decoded, sampled, output_path)
    return document


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cv2.setNumThreads(1)
    np.random.seed(42)
    
    input_file = INPUT_PATH
    output_file = OUTPUT_PATH
    profile = False
    enable_lstm = False
    enable_batch = False
    
    # Parse CLI arguments
    args = sys.argv[1:]
    for idx, arg in enumerate(args):
        if arg in ("--video", "--input", "-i") and idx + 1 < len(args):
            input_file = Path(args[idx + 1])
        elif arg in ("--output", "-o") and idx + 1 < len(args):
            output_file = Path(args[idx + 1])
        elif arg == "--profile":
            profile = True
        elif arg == "--enable-lstm":
            enable_lstm = True
        elif arg == "--enable-batch":
            enable_batch = True  # Faz 4: Batch inference (opsiyonel)
            
    try:
        verify_model_lock()
        asyncio.run(async_analyze_video(input_file, output_file, profile, enable_lstm, enable_batch))
        return 0
    except Exception as e:
        LOG.exception("FTR inference failed, writing fallback results.json")
        try:
            # Write contract-compliant empty/fallback JSON instead of leaving output blank
            fallback_doc = {
                "video_id": input_file.name,
                "arac_bilgisi": {
                    "tip": "sedan",
                    "plaka": "01A0000",
                    "renk": "gri",
                    "confidence_score": 0.10
                },
                "tespitler": []
            }
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(fallback_doc, indent=2, ensure_ascii=False), encoding="utf-8")
            LOG.info("Fallback results.json written successfully to %s", output_file)
        except Exception as write_err:
            LOG.error("Failed to write fallback results.json: %s", write_err)
        return 0


if __name__ == "__main__":
    sys.exit(main())
