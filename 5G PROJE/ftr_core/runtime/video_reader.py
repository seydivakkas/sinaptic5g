# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Generator

logger = logging.getLogger("sinaptic5g.ftr.video_reader")

class VideoReader:
    """Handles cv2.VideoCapture opening, probing, and sequential frame retrieval."""
    
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.cap = None
        self.fps = 30.0
        self.frame_count = 0
        self.duration = 0.0
        self.width = 0
        self.height = 0
        self._open()

    def _open(self):
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.file_path}")
        self.cap = cv2.VideoCapture(str(self.file_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.file_path}")
        
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0
        self.duration = self.frame_count / self.fps if self.frame_count > 0 else 0.0
        logger.info("Video opened successfully. FPS=%.2f, Frames=%d, Duration=%.2fs, Size=%dx%d",
                    self.fps, self.frame_count, self.duration, self.width, self.height)

    def read_frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Generator that yields (frame_index, timestamp_seconds, frame_bgr)"""
        frame_idx = 0
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                frame_idx += 1
                timestamp = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if not np.isfinite(timestamp) or timestamp < 0:
                    timestamp = (frame_idx - 1) / self.fps
                yield frame_idx, timestamp, frame
        finally:
            self.release()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info("Video reader capture interface released.")
