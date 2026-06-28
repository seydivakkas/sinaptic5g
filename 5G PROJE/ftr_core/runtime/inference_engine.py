# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any

logger = logging.getLogger("sinaptic5g.ftr.inference_engine")

class InferenceEngine:
    """Runs custom driver actions and coco vehicle detection models."""
    
    def __init__(self,
                 detector_session: Any,
                 coco_session: Any,
                 class_map: Dict[int, str],
                 thresholds: Dict[str, float],
                 default_threshold: float = 0.35,
                 model_size: int = 640):
        self.det_sess = detector_session
        self.coco_sess = coco_session
        self.class_map = class_map
        self.thresholds = thresholds
        self.default_thresh = default_threshold
        self.model_size = model_size

    def _decode_yolo_bbox(self, cx: float, cy: float, bw: float, bh: float, w_orig: int, h_orig: int) -> Tuple[int, int, int, int]:
        if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.0:
            scale_x, scale_y = w_orig, h_orig
        else:
            scale_x, scale_y = w_orig / self.model_size, h_orig / self.model_size

        x1 = max(0, int((cx - bw / 2) * scale_x))
        y1 = max(0, int((cy - bh / 2) * scale_y))
        x2 = min(w_orig, int((cx + bw / 2) * scale_x))
        y2 = min(h_orig, int((cy + bh / 2) * scale_y))
        return x1, y1, x2, y2

    def _valid_bbox(self, bbox: Tuple[int, int, int, int], w_orig: int, h_orig: int, min_ratio: float = 0.00005) -> bool:
        x1, y1, x2, y2 = bbox
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w < 8 or h < 8:
            return False
        return (w * h) >= (w_orig * h_orig * min_ratio)

    def detect_vehicles(self, frame: np.ndarray) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Detects vehicles using COCO model. Returns (vehicle_detections, cabin_coco_detections)."""
        if self.coco_sess is None:
            return [], []
            
        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, (self.model_size, self.model_size), interpolation=cv2.INTER_LINEAR)
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        
        input_name = self.coco_sess.get_inputs()[0].name
        raw = np.asarray(self.coco_sess.run(None, {input_name: blob})[0])
        preds = raw[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        # Vehicle mappings (YOLO COCO ids: 2 = car, 5 = bus, 7 = truck)
        COCO_VEHICLE_MAP = {2: "sedan", 5: "minibus", 7: "kamyon"}
        COCO_CABIN_CLASSES = {0, 73} # 0 = person, 73 = laptop
        
        boxes, scores, types = [], [], []
        cabin_dets = []
        
        for row in preds:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            cx, cy, bw, bh = map(float, row[:4])
            bbox = self._decode_yolo_bbox(cx, cy, bw, bh, w_orig, h_orig)
            
            # Cabin checks
            if class_id in COCO_CABIN_CLASSES and confidence >= 0.35 and self._valid_bbox(bbox, w_orig, h_orig, min_ratio=0.00005):
                cabin_dets.append({"bbox": bbox, "class_id": class_id, "confidence": confidence})
                
            vehicle_type = COCO_VEHICLE_MAP.get(class_id)
            if vehicle_type is None or confidence < 0.35:
                continue
            if not self._valid_bbox(bbox, w_orig, h_orig):
                continue
                
            boxes.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
            scores.append(confidence)
            types.append(vehicle_type)

        if not boxes:
            return [], cabin_dets
            
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.35, 0.45)
        
        # Simple detection mock wrapper
        class VehicleDetection:
            def __init__(self, bbox, vtype, conf):
                self.bbox = bbox
                self.vehicle_type = vtype
                self.confidence = conf
                
        vehicle_results = [
            VehicleDetection(
                (boxes[idx][0], boxes[idx][1], boxes[idx][0] + boxes[idx][2], boxes[idx][1] + boxes[idx][3]),
                types[idx],
                scores[idx]
            )
            for idx in np.asarray(indices).reshape(-1).tolist()
        ]
        return vehicle_results, cabin_dets

    def detect_driver_actions(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detects custom driver behaviors and plates. Returns list of detection dicts."""
        if self.det_sess is None:
            return []
            
        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, (self.model_size, self.model_size), interpolation=cv2.INTER_LINEAR)
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        
        input_name = self.det_sess.get_inputs()[0].name
        raw = np.asarray(self.det_sess.run(None, {input_name: blob})[0])
        preds = raw[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes, scores, class_ids = [], [], []
        
        for row in preds:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            
            class_name = self.class_map.get(class_id, "")
            thresh = self.thresholds.get(class_name, self.default_thresh)
            if confidence < thresh:
                continue
                
            cx, cy, bw, bh = map(float, row[:4])
            bbox = self._decode_yolo_bbox(cx, cy, bw, bh, w_orig, h_orig)
            
            min_ratio = 0.00002 if class_name == "license_plate" else 0.00005
            if not self._valid_bbox(bbox, w_orig, h_orig, min_ratio=min_ratio):
                continue
                
            boxes.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.35, 0.45)
        results = []
        for idx in np.asarray(indices).reshape(-1).tolist():
            cid = class_ids[idx]
            results.append({
                "class_id": cid,
                "class_name": self.class_map[cid],
                "confidence": scores[idx],
                "bbox": [boxes[idx][0], boxes[idx][1], boxes[idx][0] + boxes[idx][2], boxes[idx][1] + boxes[idx][3]]
            })
        return results
