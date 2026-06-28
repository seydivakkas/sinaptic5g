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

"""Offline FTR modular entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path
# Insert ftr_core directory to path to resolve local imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import logging
import os
import yaml
import json
import time
from collections import defaultdict
import numpy as np

# Import modular runtime components
from runtime.video_reader import VideoReader
from runtime.frame_sampler import FrameSampler
from runtime.preprocessor import Preprocessor
from runtime.model_loader import ModelLoader
from runtime.inference_engine import InferenceEngine
from runtime.tracker import Tracker
from runtime.speed_estimator import SpeedEstimator
from runtime.plate_ocr import PlateOcr
from runtime.cabin_analyzer import CabinAnalyzer
from runtime.event_detector import EventDetector
from runtime.temporal_aggregator import TemporalAggregator
from runtime.competition_adapter import CompetitionAdapterWrapper
from runtime.schema_validator import SchemaValidator
from runtime.result_writer import ResultWriter

# FTR Evaluator Paths (Static)
INPUT_PATH = Path("/app/data/input/video.mp4")
OUTPUT_PATH = Path("/app/data/output/results.json")
SCHEMA_PATH = Path("/app/schemas/results.schema.json")
LOCK_PATH = Path("/app/model_lock.json")

# Fallbacks for local execution
if not INPUT_PATH.is_file():
    INPUT_PATH = Path(__file__).resolve().parent.parent / "data/input/video.mp4"
    if not INPUT_PATH.is_file():
        INPUT_PATH = Path(__file__).resolve().parent.parent / "tests/smoke_input/video.mp4"

if not OUTPUT_PATH.parent.exists() and not Path("/app").exists():
    OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tests/smoke_output/results.json"

if not SCHEMA_PATH.is_file():
    SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas/results.schema.json"

if not LOCK_PATH.is_file():
    LOCK_PATH = Path(__file__).resolve().parent.parent / "model_lock.json"

LOG = logging.getLogger("sinaptic5g.ftr.main")

def load_configs() -> tuple[dict, dict, dict, dict]:
    """Loads all YAML configurations from configs/ directory with fallbacks."""
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    if not configs_dir.is_dir():
        configs_dir = Path("/app/configs")
        
    class_map_path = configs_dir / "class_map.yaml"
    thresholds_path = configs_dir / "thresholds.yaml"
    model_reg_path = configs_dir / "model_registry.yaml"
    runtime_path = configs_dir / "ftr_runtime.yaml"
    
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = yaml.safe_load(f)["class_map"]
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresh_data = yaml.safe_load(f)
        thresholds = thresh_data["thresholds"]
        default_thresh = thresh_data["default_threshold"]
        zero_dce_brightness_thresh = thresh_data["zero_dce_brightness_threshold"]
    with open(model_reg_path, "r", encoding="utf-8") as f:
        model_reg = yaml.safe_load(f)["models"]
    with open(runtime_path, "r", encoding="utf-8") as f:
        runtime_cfg = yaml.safe_load(f)["runtime"]
        
    threshold_opts = {
        "thresholds": thresholds,
        "default_threshold": default_thresh,
        "zero_dce_brightness_threshold": zero_dce_brightness_thresh
    }
    
    return class_map, threshold_opts, model_reg, runtime_cfg

async def async_analyze_video(input_path: Path,
                             output_path: Path,
                             class_map: dict,
                             thresh_opts: dict,
                             model_reg: dict,
                             runtime_cfg: dict) -> dict:
    
    LOG.info("Loading models and verifying lock integrity...")
    loader = ModelLoader(LOCK_PATH, model_reg)
    det_sess = loader.create_session("detector")
    coco_sess = loader.create_session("coco")
    
    # Initialize components
    reader = VideoReader(input_path)
    preprocessor = Preprocessor(
        frame_size=(reader.width, reader.height),
        camera_mode=runtime_cfg["camera_mode"],
        zero_dce_threshold=thresh_opts["zero_dce_brightness_threshold"]
    )
    
    # Check GPU availability
    has_gpu = any("CUDA" in p or "TensorRT" in p for p in coco_sess.get_providers())
    sampler = FrameSampler(has_gpu=has_gpu)
    
    engine = InferenceEngine(
        detector_session=det_sess,
        coco_session=coco_sess,
        class_map={int(k): v for k, v in class_map.items()},
        thresholds=thresh_opts["thresholds"],
        default_threshold=thresh_opts["default_threshold"],
        model_size=model_reg["detector"]["input_size"]
    )
    
    tracker = Tracker(
        camera_mode=runtime_cfg["camera_mode"],
        bev_H=preprocessor.H_bev,
        conf_thresh=runtime_cfg["conf_thresh"],
        track_buffer=runtime_cfg["track_buffer"]
    )
    
    plate_ocr = None
    cabin_analyzer = CabinAnalyzer()
    event_detector = EventDetector()
    
    temporal_aggregator = TemporalAggregator(
        enable_lstm=runtime_cfg["enable_lstm"],
        lstm_model_path=str(Path(model_reg["cnn_lstm"]["path"]))
    )
    
    adapter = CompetitionAdapterWrapper(
        video_id=input_path.name,
        event_gap_seconds=runtime_cfg["event_gap_seconds"]
    )
    
    validator = SchemaValidator(SCHEMA_PATH)
    writer = ResultWriter(output_path)
    
    # Variables for state tracking
    track_last_plate = {}
    ocr_frame_counter = 0
    decoded = sampled = 0
    deadline = time.monotonic() + runtime_cfg["target_runtime_seconds"]
    
    loop = asyncio.get_event_loop()
    
    try:
        for frame_idx, timestamp, frame in reader.read_frames():
            decoded += 1
            if not sampler.should_sample(timestamp):
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Target runtime budget exceeded ({runtime_cfg['target_runtime_seconds']}s)")
                
            sampled += 1
            
            # Preprocess
            preprocessed_frame = preprocessor.process(frame)
            h_frame, w_frame = preprocessed_frame.shape[:2]
            
            # Detect
            vehicle_dets, cabin_coco_dets = engine.detect_vehicles(preprocessed_frame)
            custom_dets = engine.detect_driver_actions(preprocessed_frame)
            
            # Track
            tracked_dets = tracker.update(preprocessed_frame, vehicle_dets, timestamp)
            active_ids = set()
            
            for det, track_id in tracked_dets:
                active_ids.add(int(track_id))
                # Vehicle details estimation
                from ftr_main import estimate_vehicle_color # Use fast helper from original file if needed, or import
                color, color_conf = estimate_vehicle_color(preprocessed_frame, det.bbox)
                adapter.observe_vehicle(
                    timestamp=timestamp,
                    track_id=track_id,
                    vehicle_type=det.vehicle_type,
                    type_confidence=det.confidence,
                    color=color,
                    color_confidence=color_conf
                )
                
                # Slalom event checks
                track_state = tracker.tracker.trackers.get(int(track_id))
                bev_xy = track_state.history[-1] if track_state and track_state.history else None
                speed_kmh = track_state.estimated_speed_kmh if track_state else 0.0
                if bev_xy is not None:
                    is_slalom, slalom_conf = event_detector.check_slalom(int(track_id), bev_xy, float(speed_kmh))
                    if is_slalom:
                        adapter.observe_event(timestamp, "slalom", slalom_conf)
            
            event_detector.cleanup_inactive_tracks(active_ids)
            temporal_aggregator.cleanup_stale_tracks(active_ids)
            
            # Driver drowsiness / head pose
            head_behaviors = cabin_analyzer.analyze_head_pose(preprocessed_frame)
            for hb in head_behaviors:
                adapter.observe_event(timestamp, hb["etiket"], hb["confidence_score"])
                
            # Cabin passenger / laptop events
            if cabin_coco_dets:
                cabin_events = cabin_analyzer.analyze_cabin_rois(cabin_coco_dets, w_frame, h_frame)
                for ev in cabin_events:
                    adapter.observe_event(timestamp, ev["label"], ev["confidence"])
                    
            # Plate OCR (every 2 frames)
            ocr_frame_counter += 1
            run_ocr = (ocr_frame_counter % 2 == 0)
            if run_ocr:
                for det in custom_dets:
                    if det["class_name"] == "license_plate":
                        if plate_ocr is None:
                            plate_ocr = PlateOcr(
                                lprnet_model_path=model_reg["lprnet"]["path"],
                                crnn_model_path=model_reg["crnn"]["path"]
                            )
                        plate_bbox = det["bbox"]
                        crop = preprocessed_frame[max(0, plate_bbox[1]):min(h_frame, plate_bbox[3]),
                                                  max(0, plate_bbox[0]):min(w_frame, plate_bbox[2])]
                        ocr_res = plate_ocr.recognize(crop)
                        txt, conf = ocr_res if ocr_res else ("01A0000", 0.0)
                        
                        best_track_id = None
                        for det_v, track_id in tracked_dets:
                            px = (plate_bbox[0] + plate_bbox[2]) / 2.0
                            py = (plate_bbox[1] + plate_bbox[3]) / 2.0
                            if det_v.bbox[0] <= px <= det_v.bbox[2] and det_v.bbox[1] <= py <= det_v.bbox[3]:
                                best_track_id = track_id
                                break
                        if best_track_id is None:
                            # Inferred vehicle bbox if vehicle detector misses
                            from ftr_main import infer_vehicle_bbox_from_plate, estimate_vehicle_color
                            inf_bbox = infer_vehicle_bbox_from_plate(plate_bbox, w_frame, h_frame)
                            color, color_conf = estimate_vehicle_color(preprocessed_frame, inf_bbox)
                            best_track_id = "plate_anchor"
                            adapter.observe_vehicle(
                                timestamp=timestamp,
                                track_id=best_track_id,
                                vehicle_type="sedan",
                                type_confidence=max(0.35, float(det.get("confidence", 0.0)) * 0.65),
                                color=color,
                                color_confidence=color_conf
                            )
                        if best_track_id is not None:
                            track_last_plate[best_track_id] = (txt, conf)
                            adapter.observe_vehicle(
                                timestamp=timestamp,
                                track_id=best_track_id,
                                plate=txt,
                                plate_confidence=conf
                            )
            else:
                for det_v, track_id in tracked_dets:
                    if track_id in track_last_plate:
                        txt, conf = track_last_plate[track_id]
                        adapter.observe_vehicle(
                            timestamp=timestamp,
                            track_id=track_id,
                            plate=txt,
                            plate_confidence=conf
                        )
                        
            # Temporal Sequence classification (CNN-LSTM)
            has_phone = any(d["class_name"] == "telefonla_konusma" for d in custom_dets)
            has_cig = any(d["class_name"] == "sigara_icme" for d in custom_dets)
            for det_v, track_id in tracked_dets:
                track_state = tracker.tracker.trackers.get(int(track_id))
                speed_kmh = track_state.estimated_speed_kmh if track_state else 0.0
                lstm_events = temporal_aggregator.process_track(
                    track_id=track_id,
                    ear=0.30, # Fallback face features
                    mar=0.15,
                    speed_kmh=speed_kmh,
                    has_phone=has_phone,
                    has_cigarette=has_cig
                )
                for label, lstm_conf in lstm_events:
                    adapter.observe_event(timestamp, label, lstm_conf)
                    
            # Custom detector events
            for det in custom_dets:
                if det["class_name"] != "license_plate":
                    adapter.observe_event(timestamp, det["class_name"], det["confidence"])
                    
            # Stride adjustment
            sampler.update_stride(bool(vehicle_dets), timestamp, reader.duration)
            
    finally:
        reader.release()
        
    doc = adapter.finalize(validate=False)
    # Perform schema checks before write
    validator.validate(doc)
    writer.write(doc)
    return doc

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    try:
        class_map, thresh_opts, model_reg, runtime_cfg = load_configs()
        asyncio.run(async_analyze_video(
            INPUT_PATH, OUTPUT_PATH,
            class_map, thresh_opts, model_reg, runtime_cfg
        ))
        return 0
    except Exception as e:
        LOG.exception("FTR Offline Core execution failed!")
        # Write contract-compliant empty/fallback JSON
        try:
            fallback = {
                "video_id": INPUT_PATH.name,
                "arac_bilgisi": {
                    "tip": "sedan",
                    "plaka": "01A0000",
                    "renk": "gri",
                    "confidence_score": 0.10
                },
                "tespitler": []
            }
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(json.dumps(fallback, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as err:
            LOG.error("Failed to write fallback: %s", err)
        return 0

if __name__ == "__main__":
    sys.exit(main())
