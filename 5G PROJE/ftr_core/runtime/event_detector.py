# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import logging
from typing import Dict, Tuple, List, Any

logger = logging.getLogger("sinaptic5g.ftr.event_detector")

class EventDetector:
    """Detects complex driving events like slalom maneuvers and rules violations."""
    
    def __init__(self):
        from src.slalom_detector import SlalomDetector
        # History length=30, changes=3, min speed=10km/h
        self.slalom_detector = SlalomDetector(history_len=30, min_changes=3, min_speed_kmh=10.0, min_shift=0.5)

    def check_slalom(self, track_id: int, bev_xy: Tuple[float, float], speed_kmh: float) -> Tuple[bool, float]:
        """Runs slalom detector on vehicle's BEV coordinates. Returns (is_slalom, confidence)."""
        return self.slalom_detector.update(track_id, bev_xy, speed_kmh)

    def cleanup_inactive_tracks(self, active_track_ids: List[int]):
        """Cleans up slalom track history of inactive tracked targets."""
        self.slalom_detector.cleanup(active_track_ids)
