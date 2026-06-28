# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import numpy as np
from runtime.frame_sampler import FrameSampler
from runtime.preprocessor import Preprocessor
from runtime.speed_estimator import SpeedEstimator

def test_frame_sampler_stride_computation():
    sampler = FrameSampler(has_gpu=False)
    
    # Initial state should require sampling at t=0
    assert sampler.should_sample(0.0) is True
    
    # Processed frame at t=0, detections=False -> stride=1.5 on CPU
    sampler.update_stride(has_detections=False, timestamp=0.0)
    assert sampler.should_sample(0.5) is False
    assert sampler.should_sample(1.5) is True

def test_speed_estimator_calculation():
    estimator = SpeedEstimator(fps=30.0)
    
    prev_bev = (1.0, 1.0)
    curr_bev = (1.0, 6.0) # Moved 5 meters
    speed = estimator.estimate_speed_kmh(prev_bev, curr_bev, time_delta=1.0)
    
    # 5 m/s * 3.6 = 18.0 km/h
    assert abs(speed - 18.0) < 1e-4
