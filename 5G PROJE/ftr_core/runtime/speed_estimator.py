# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import numpy as np
from typing import Tuple

class SpeedEstimator:
    """Calculates physical vehicle speed from consecutive BEV coordinates and timestamps."""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def estimate_speed_kmh(self,
                            prev_coord: Tuple[float, float],
                            curr_coord: Tuple[float, float],
                            time_delta: float) -> float:
        """Estimates speed based on simple euclidean distance in BEV space and elapsed time."""
        if time_delta <= 0:
            return 0.0
            
        dx = curr_coord[0] - prev_coord[0]
        dy = curr_coord[1] - prev_coord[1]
        distance_meters = np.sqrt(dx*dx + dy*dy)
        
        # meters per second to km/h conversion factor = 3.6
        speed_mps = distance_meters / time_delta
        return float(speed_mps * 3.6)
