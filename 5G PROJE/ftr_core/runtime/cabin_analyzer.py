# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import logging
import cv2
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("sinaptic5g.ftr.cabin_analyzer")

class CabinAnalyzer:
    """Manages driver head-pose tracking (MediaPipe) and cabin person/object spatial relations."""
    
    def __init__(self):
        # Initialize FaceMesh
        self.face_mesh = None
        self.mp_lib = None
        self._init_facemesh()
        
        # Core detectors
        from src.head_pose_detector import HeadPoseDetector
        from src.cabin_roi_detector import CabinRoiDetector
        self.head_pose_detector = HeadPoseDetector(buffer_size=8, trigger_count=5)
        self.cabin_detector = CabinRoiDetector()

    def _init_facemesh(self):
        try:
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_lib = mp
            logger.info("MediaPipe FaceMesh initialised in CabinAnalyzer.")
        except Exception as e:
            logger.warning("MediaPipe FaceMesh initialization failed: %s", e)

    def analyze_head_pose(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Extracts head-pose behaviors (e.g. etrafa_bakinma, esneme) using FaceMesh."""
        if self.face_mesh is None:
            return []
            
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        behaviors = []
        if results.multi_face_landmarks:
            res = self.head_pose_detector.process_landmarks(results.multi_face_landmarks[0], w, h)
            if res.get("behavior") is not None:
                behaviors.append(res["behavior"])
        return behaviors

    def analyze_cabin_rois(self, coco_detections: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
        """Maps COCO person/laptop coordinates to cabin/passenger events."""
        return self.cabin_detector.process_detections(coco_detections, width, height)
