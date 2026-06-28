from dataclasses import dataclass

import numpy as np

from src.tracking_pipeline import AdaptiveKalmanTracker, SinapticTracker


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float = 0.9


def test_tracker_keeps_id_for_overlapping_motion():
    tracker = SinapticTracker(bev_H=np.eye(3, dtype=np.float32))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    first = tracker.update(frame, [Detection((50, 80, 150, 180))], 0.0)
    second = tracker.update(frame, [Detection((56, 80, 156, 180))], 0.1)

    assert first[0][1] == second[0][1]
    assert len(tracker.trackers) == 1


def test_speed_uses_timestamps_instead_of_fixed_fps():
    state = AdaptiveKalmanTracker(1, np.array([0.0, 0.0]), init_timestamp=0.0)
    for index in range(1, 5):
        timestamp = index * 0.5
        state.predict(timestamp)
        state.update(np.array([float(index), 0.0]), timestamp=timestamp)

    assert 7.0 < state.estimated_speed_kmh < 7.5
