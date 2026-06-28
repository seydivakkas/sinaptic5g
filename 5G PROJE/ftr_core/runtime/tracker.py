# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import logging
import numpy as np
from typing import List, Tuple, Any

logger = logging.getLogger("sinaptic5g.ftr.tracker")

class Tracker:
    """Stateful vehicle tracker using BoTSORT (Kalman tracking + Optical Flow CMC)."""
    
    def __init__(self,
                 camera_mode: str = "front",
                 bev_H: np.ndarray = None,
                 conf_thresh: float = 0.45,
                 track_buffer: int = 45):
        from src.tracking_pipeline import SinapticTracker
        self.tracker = SinapticTracker(
            camera_mode=camera_mode,
            bev_H=bev_H,
            conf_thresh=conf_thresh,
            track_buffer=track_buffer
        )

    def update(self, frame: np.ndarray, detections: List[Any], timestamp: float) -> List[Tuple[Any, int]]:
        """Updates tracker state and returns list of (detection, track_id) tuples."""
        return self.tracker.update(frame, detections, timestamp)

    def get_track_history(self, track_id: int) -> List[Tuple[float, float]]:
        """Returns the BEV coordinate history of the specified track_id."""
        track_state = self.tracker.trackers.get(track_id)
        if track_state and track_state.history:
            return track_state.history
        return []

    def get_estimated_speed(self, track_id: int) -> float:
        """Returns the estimated speed in km/h for the track_id."""
        track_state = self.tracker.trackers.get(track_id)
        if track_state:
            return float(track_state.estimated_speed_kmh)
        return 0.0
