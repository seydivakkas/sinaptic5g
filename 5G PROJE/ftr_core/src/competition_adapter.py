"""Track-level aggregation and the sole competition serializer boundary."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .competition_contract import (
    VEHICLE_COLORS,
    VEHICLE_TYPES,
    atomic_write_results,
    clamp_confidence,
    normalize_label,
    normalize_plate,
    validate_results,
)


VEHICLE_CLASS_MAP = {
    "car": "sedan",
    "vehicle": "sedan",
    "togg": "sedan",
    "truck": "kamyon",
    "bus": "minibus",
    "van": "panelvan",
}


def _value(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


@dataclass
class VehicleTrack:
    track_id: str
    first_timestamp: float
    last_timestamp: float
    types: list[tuple[str, float]] = field(default_factory=list)
    colors: list[tuple[str, float]] = field(default_factory=list)
    plates: list[tuple[str, float]] = field(default_factory=list)

    def observe(
        self,
        timestamp: float,
        vehicle_type: str | None = None,
        type_confidence: float = 0.0,
        color: str | None = None,
        color_confidence: float = 0.0,
        plate: str | None = None,
        plate_confidence: float = 0.0,
    ) -> None:
        self.first_timestamp = min(self.first_timestamp, timestamp)
        self.last_timestamp = max(self.last_timestamp, timestamp)
        if vehicle_type in VEHICLE_TYPES:
            self.types.append((vehicle_type, clamp_confidence(type_confidence)))
        if color in VEHICLE_COLORS:
            self.colors.append((color, clamp_confidence(color_confidence)))
        normalized_plate = normalize_plate(plate)
        if normalized_plate:
            self.plates.append((normalized_plate, clamp_confidence(plate_confidence)))

    @property
    def duration(self) -> float:
        return max(0.0, self.last_timestamp - self.first_timestamp)

    @property
    def valid_plate(self) -> bool:
        return bool(self.plates)

    @staticmethod
    def _weighted_choice(values: list[tuple[str, float]], fallback: str) -> tuple[str, float]:
        if not values:
            return fallback, 0.0
        totals: dict[str, float] = defaultdict(float)
        counts: Counter[str] = Counter()
        for label, confidence in values:
            totals[label] += max(confidence, 1e-9)
            counts[label] += 1
        label = max(totals, key=lambda item: (totals[item], counts[item], item))
        scores = sorted((score for value, score in values if value == label), reverse=True)
        return label, clamp_confidence(statistics.fmean(scores[:5]))

    def summary(self) -> dict[str, Any]:
        vehicle_type, type_conf = self._weighted_choice(self.types, "sedan")
        color, color_conf = self._weighted_choice(self.colors, "gri")
        plate, plate_conf = self._weighted_choice(self.plates, "01A0000")
        return {
            "tip": vehicle_type,
            "plaka": plate,
            "renk": color,
            "confidence_score": clamp_confidence(min(type_conf, plate_conf, color_conf)),
        }

    def selection_confidence(self) -> float:
        scores = [score for _, score in (*self.types, *self.colors, *self.plates)]
        return statistics.fmean(scores) if scores else 0.0


@dataclass
class EventSegment:
    category: str
    label: str
    started_at: float
    ended_at: float
    observations: list[tuple[float, float]] = field(default_factory=list)

    def append(self, timestamp: float, confidence: float) -> None:
        self.ended_at = max(self.ended_at, timestamp)
        self.observations.append((timestamp, clamp_confidence(confidence)))

    def public_value(self) -> dict[str, Any]:
        ranked = sorted(self.observations, key=lambda item: item[1], reverse=True)
        best_timestamp = ranked[0][0]
        top_scores = [item[1] for item in ranked[:5]]
        return {
            "zaman_saniye": round(max(0.0, best_timestamp), 3),
            "kategori": self.category,
            "etiket": self.label,
            "confidence_score": clamp_confidence(statistics.fmean(top_scores)),
        }


class CompetitionAdapter:
    """Aggregate rich internal detections into the strict public contract."""

    def __init__(self, video_id: str = "video.mp4", event_gap_seconds: float = 1.0):
        self.video_id = Path(video_id).name
        self.event_gap_seconds = max(0.0, event_gap_seconds)
        self._tracks: dict[str, VehicleTrack] = {}
        self._segments: list[EventSegment] = []
        self._open_segments: dict[str, EventSegment] = {}

    def observe_vehicle(
        self,
        *,
        timestamp: float,
        track_id: str | int = "primary",
        vehicle_type: str | None = None,
        type_confidence: float = 0.0,
        color: str | None = None,
        color_confidence: float = 0.0,
        plate: str | None = None,
        plate_confidence: float = 0.0,
    ) -> None:
        key = str(track_id)
        track = self._tracks.setdefault(key, VehicleTrack(key, timestamp, timestamp))
        track.observe(
            timestamp,
            vehicle_type,
            type_confidence,
            color,
            color_confidence,
            plate,
            plate_confidence,
        )

    def observe_event(self, timestamp: float, label: str, confidence: float) -> None:
        normalized = normalize_label(label)
        if normalized is None:
            return
        category, public_label = normalized
        current = self._open_segments.get(public_label)
        if current is None or timestamp - current.ended_at > self.event_gap_seconds:
            if current is not None:
                self._segments.append(current)
            current = EventSegment(category, public_label, timestamp, timestamp)
            self._open_segments[public_label] = current
        current.append(timestamp, confidence)

    def process_frame(self, analysis: Any) -> None:
        timestamp = float(_value(analysis, "timestamp", default=0.0) or 0.0)
        track_id = _value(analysis, "track_id", default="primary")

        explicit_type = _value(analysis, "vehicle_type", default=None)
        explicit_color = _value(analysis, "vehicle_color", "color", default=None)
        explicit_plate = _value(analysis, "license_plate_text", "plate", default=None)
        if explicit_type or explicit_color or explicit_plate:
            self.observe_vehicle(
                timestamp=timestamp,
                track_id=track_id,
                vehicle_type=explicit_type,
                type_confidence=_value(analysis, "vehicle_type_confidence", default=0.0),
                color=explicit_color,
                color_confidence=_value(analysis, "vehicle_color_confidence", default=0.0),
                plate=explicit_plate,
                plate_confidence=_value(analysis, "license_plate_confidence", default=0.0),
            )

        for detection in _value(analysis, "detections", default=[]) or []:
            class_name = str(_value(detection, "class_name", "class", default="")).lower()
            confidence = _value(detection, "confidence", "score", default=0.0)
            detection_track = _value(detection, "track_id", default=track_id)
            vehicle_type = _value(detection, "vehicle_type", default=None)
            vehicle_type = vehicle_type or VEHICLE_CLASS_MAP.get(class_name)
            if vehicle_type:
                self.observe_vehicle(
                    timestamp=timestamp,
                    track_id=detection_track,
                    vehicle_type=vehicle_type,
                    type_confidence=confidence,
                    color=_value(detection, "color", default=None),
                    color_confidence=_value(detection, "color_confidence", default=0.0),
                    plate=_value(detection, "plate", default=None),
                    plate_confidence=_value(detection, "plate_confidence", default=0.0),
                )
            if class_name in {"phone", "cigarette", "toy", "laptop", "computer"}:
                alias = {
                    "phone": "telefonla_konusma",
                    "cigarette": "sigara_icme",
                    "toy": "teknocan",
                    "laptop": "bilgisayar",
                    "computer": "bilgisayar",
                }[class_name]
                self.observe_event(timestamp, alias, confidence)

        behavior = _value(analysis, "behavior_class", default=None)
        behavior_confidence = _value(analysis, "behavior_conf", "behavior_confidence", default=0.0)
        if behavior and str(behavior).lower() != "normal_surus":
            self.observe_event(timestamp, str(behavior), behavior_confidence)

    def _primary_track(self) -> VehicleTrack:
        if not self._tracks:
            fallback = VehicleTrack("fallback", 0.0, 0.0)
            fallback.observe(0.0, "sedan", 0.0, "gri", 0.0, "01A0000", 0.0)
            return fallback
        return max(
            self._tracks.values(),
            key=lambda track: (
                int(track.valid_plate),
                track.duration,
                track.selection_confidence(),
                -track.first_timestamp,
                track.track_id,
            ),
        )

    def _merge_duplicate_plates(self) -> None:
        """Merge different vehicle tracks that share the same normalized license plate."""
        from collections import defaultdict
        
        plate_to_tracks = defaultdict(list)
        for track_id, track in self._tracks.items():
            plate, _ = track._weighted_choice(track.plates, "")
            if plate and plate != "01A0000":
                plate_to_tracks[plate].append(track)
                
        for plate, tracks in plate_to_tracks.items():
            if len(tracks) > 1:
                # Merge into the primary track (the one with longest duration or highest confidence)
                primary = max(tracks, key=lambda t: (t.duration, t.selection_confidence()))
                for other in tracks:
                    if other.track_id == primary.track_id:
                        continue
                    # Combine all observations into primary track
                    primary.types.extend(other.types)
                    primary.colors.extend(other.colors)
                    primary.plates.extend(other.plates)
                    primary.first_timestamp = min(primary.first_timestamp, other.first_timestamp)
                    primary.last_timestamp = max(primary.last_timestamp, other.last_timestamp)
                    # Remove the merged other track
                    self._tracks.pop(other.track_id, None)

    def _gate_short_tracks(self) -> None:
        """Filter out very short / low confidence tracks to reduce false positive vehicle detections."""
        to_remove = []
        for track_id, track in self._tracks.items():
            # If the track duration is less than 0.4 seconds (10 frames @ 25 fps) and has no plates, gate it
            if track.duration < 0.4 and not track.plates:
                to_remove.append(track_id)
        for track_id in to_remove:
            self._tracks.pop(track_id, None)

    def finalize(self, validate: bool = True) -> dict[str, Any]:
        self._merge_duplicate_plates()
        self._gate_short_tracks()
        segments = list(self._segments) + list(self._open_segments.values())
        document = {
            "video_id": self.video_id,
            "arac_bilgisi": self._primary_track().summary(),
            "tespitler": sorted(
                (segment.public_value() for segment in segments),
                key=lambda item: (item["zaman_saniye"], item["kategori"], item["etiket"]),
            ),
        }
        if validate:
            validate_results(document)
        return document

    def write(self, output_path: str | Path) -> Path:
        return atomic_write_results(self.finalize(validate=False), output_path)

