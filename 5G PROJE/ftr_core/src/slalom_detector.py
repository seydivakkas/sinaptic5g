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

"""F4 Task: Slalom detector using BEV coordinate tracking history."""

import numpy as np
from collections import deque

class SlalomDetector:
    """Detects slalom anomaly based on BEV coordinates trajectory."""

    def __init__(self, history_len: int = 30, min_changes: int = 3, min_speed_kmh: float = 10.0, min_shift: float = 0.5):
        self.history_len = history_len
        self.min_changes = min_changes
        self.min_speed_kmh = min_speed_kmh
        self.min_shift = min_shift
        # Map track_id -> deque of x-coordinates
        self.track_x_history = {}

    def update(self, track_id: int, bev_xy: np.ndarray, speed_kmh: float) -> tuple[bool, float]:
        """Update track trajectory and check if it performs slalom behavior.

        Returns:
            (is_slalom, confidence)
        """
        if track_id not in self.track_x_history:
            self.track_x_history[track_id] = deque(maxlen=self.history_len)

        x_val = float(bev_xy[0])
        self.track_x_history[track_id].append(x_val)

        x_hist = list(self.track_x_history[track_id])
        if len(x_hist) < 15:
            return False, 0.0

        # Calculate max shift
        x_shift = max(x_hist) - min(x_hist)

        # Count direction changes
        diffs = [x_hist[i] - x_hist[i-1] for i in range(1, len(x_hist))]
        direction_changes = 0
        last_sign = 0
        
        for d in diffs:
            if abs(d) > 0.03:  # Noise threshold of 3cm lateral movement
                sign = 1 if d > 0 else -1
                if last_sign != 0 and sign != last_sign:
                    direction_changes += 1
                last_sign = sign

        # Check conditions
        if direction_changes >= self.min_changes and speed_kmh > self.min_speed_kmh and x_shift > self.min_shift:
            confidence = min(direction_changes / 5.0, 1.0)
            return True, confidence

        return False, 0.0

    def cleanup(self, active_ids: set[int]):
        """Remove history for inactive tracks."""
        for stale_id in list(self.track_x_history.keys()):
            if stale_id not in active_ids:
                self.track_x_history.pop(stale_id, None)
