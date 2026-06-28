"""Calibrated road-plane speed estimator for the optional live display.

Absolute km/h is never produced without a per-camera homography artifact.
The FTR public JSON does not contain speed, so this module is outside its
critical path.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


LOG = logging.getLogger(__name__)


class SpeedEstimator:
    def __init__(
        self,
        fps: float = 30.0,
        calibration_path: str | Path | None = None,
        smoothing_window: int = 5,
        **_: object,
    ):
        self.fps = fps
        self.homography: np.ndarray | None = None
        self._previous_world_point: np.ndarray | None = None
        self._previous_timestamp: float | None = None
        self._history: deque[float] = deque(maxlen=smoothing_window)
        if calibration_path:
            self.load_calibration(calibration_path)

    @property
    def calibrated(self) -> bool:
        return self.homography is not None

    def load_calibration(self, path: str | Path) -> None:
        with Path(path).open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact.get("status") != "OLCULDU":
            raise ValueError("camera calibration artifact must have status=OLCULDU")
        matrix = np.asarray(artifact["homography_image_to_road"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("homography must be a finite 3x3 matrix")
        self.homography = matrix
        LOG.info("Camera homography loaded: %s", path)

    def estimate(
        self,
        frame: np.ndarray,
        vehicle_bbox: Optional[tuple[int, int, int, int]] = None,
        timestamp: float | None = None,
    ) -> Optional[float]:
        if self.homography is None or vehicle_bbox is None:
            return None
        x1, _y1, x2, y2 = vehicle_bbox
        contact_point = np.array([[[0.5 * (x1 + x2), float(y2)]]], dtype=np.float64)
        world_point = cv2.perspectiveTransform(contact_point, self.homography)[0, 0]
        current_timestamp = float(timestamp if timestamp is not None else time.monotonic())

        if self._previous_world_point is None or self._previous_timestamp is None:
            self._previous_world_point = world_point
            self._previous_timestamp = current_timestamp
            return None
        delta_seconds = current_timestamp - self._previous_timestamp
        if delta_seconds <= 0:
            return None
        distance_meters = float(np.linalg.norm(world_point - self._previous_world_point))
        speed_kmh = distance_meters / delta_seconds * 3.6
        self._previous_world_point = world_point
        self._previous_timestamp = current_timestamp
        if not np.isfinite(speed_kmh) or speed_kmh < 0 or speed_kmh > 250:
            return None
        self._history.append(speed_kmh)
        return round(float(np.median(self._history)), 1)

    def reset(self) -> None:
        self._previous_world_point = None
        self._previous_timestamp = None
        self._history.clear()

