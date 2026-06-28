"""Sinaptic5G — Model Entegrasyonları.

Bu paket üç alt modülü kapsar:
    - detection    : YOLOv8 nesne tespiti wrapper
    - low_light    : Zero-DCE düşük ışık iyileştirme
    - temporal     : Yörünge takibi ve davranış sınıflandırma
"""

from src.models.detection import (
    DetectionResult,
    YOLODetector,
    TURKISH_TRAFFIC_CLASSES,
)
from src.models.low_light.zero_dce import (
    ZeroDCENet,
    ZeroDCELoss,
    SpatialConsistencyLoss,
    ExposureControlLoss,
    ColorConstancyLoss,
    TotalVariationLoss,
)
from src.models.temporal import (
    TrajectoryBuffer,
    TemporalAnomalyDetector,
    DrowsinessTemporalAnalyzer,
    BehaviorClassifier,
    BehaviorCategory,
    DrivingBehaviorState,
    TemporalCNNLSTM,
    FeatureExtractor,
    BEHAVIOR_CLASSES,
    FEATURE_DIM,
)

__all__ = [
    # Detection
    "DetectionResult",
    "YOLODetector",
    "TURKISH_TRAFFIC_CLASSES",
    # Low Light
    "ZeroDCENet",
    "ZeroDCELoss",
    "SpatialConsistencyLoss",
    "ExposureControlLoss",
    "ColorConstancyLoss",
    "TotalVariationLoss",
    # Temporal
    "TrajectoryBuffer",
    "TemporalAnomalyDetector",
    "DrowsinessTemporalAnalyzer",
    "BehaviorClassifier",
    "BehaviorCategory",
    "DrivingBehaviorState",
    "TemporalCNNLSTM",
    "FeatureExtractor",
    "BEHAVIOR_CLASSES",
    "FEATURE_DIM",
]
