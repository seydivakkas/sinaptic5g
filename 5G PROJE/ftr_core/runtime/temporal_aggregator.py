# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("sinaptic5g.ftr.temporal_aggregator")

class TemporalAggregator:
    """Aggregates frame-level outputs into temporal sequences using CNN-LSTM behavior models."""
    
    def __init__(self, enable_lstm: bool = False, lstm_model_path: str = "models/cnn_lstm.onnx"):
        self.enable_lstm = enable_lstm
        self.feature_extractor = None
        self.temporal_classifier = None
        self.track_feature_buffers = defaultdict(list)
        self.track_debounce_counters = defaultdict(lambda: defaultdict(int))
        
        if self.enable_lstm:
            from src.models.temporal.cnn_lstm import ONNXTemporalClassifier, FeatureExtractor
            self.feature_extractor = FeatureExtractor()
            self.temporal_classifier = ONNXTemporalClassifier(model_path=lstm_model_path)
            logger.info("Temporal CNN-LSTM aggregator enabled.")

    def process_track(self,
                      track_id: int,
                      ear: float,
                      mar: float,
                      speed_kmh: float,
                      has_phone: bool,
                      has_cigarette: bool) -> List[Tuple[str, float]]:
        """Processes track behaviors over time. Returns list of triggered (label, confidence) events."""
        if not self.enable_lstm or self.feature_extractor is None or self.temporal_classifier is None:
            return []
            
        feat = self.feature_extractor.extract(
            ear=ear,
            mar=mar,
            speed_px=0.0,
            angle_deg=0.0,
            has_phone=has_phone,
            has_cigarette=has_cigarette,
            speed_kmh=speed_kmh,
            speed_limit_kmh=50.0
        )
        
        buf = self.track_feature_buffers[track_id]
        buf.append(feat)
        if len(buf) > 16:
            buf.pop(0)

        triggered_events = []
        if len(buf) == 16:
            pred_res = self.temporal_classifier.predict(np.array(buf))
            if pred_res is not None:
                lstm_label, lstm_conf = pred_res
                if lstm_label != "normal_surus":
                    self.track_debounce_counters[track_id][lstm_label] += 1
                    if self.track_debounce_counters[track_id][lstm_label] >= 6:
                        triggered_events.append((lstm_label, lstm_conf))
                        # Reset other labels
                        for lbl in list(self.track_debounce_counters[track_id].keys()):
                            if lbl != lstm_label:
                                self.track_debounce_counters[track_id][lbl] = 0
                else:
                    for lbl in list(self.track_debounce_counters[track_id].keys()):
                        self.track_debounce_counters[track_id][lbl] = max(0, self.track_debounce_counters[track_id][lbl] - 1)
                        
        return triggered_events

    def cleanup_stale_tracks(self, active_track_ids: set[int]):
        """Clears buffers for tracks that are no longer active to prevent memory leaks."""
        stale_ids = [k for k in self.track_debounce_counters if k not in active_track_ids]
        for k in stale_ids:
            del self.track_debounce_counters[k]
        stale_ids = [k for k in self.track_feature_buffers if k not in active_track_ids]
        for k in stale_ids:
            del self.track_feature_buffers[k]
