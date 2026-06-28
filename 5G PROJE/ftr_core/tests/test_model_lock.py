# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import json
from pathlib import Path

def test_model_lock_verification():
    lock_path = Path(__file__).resolve().parent.parent.parent / "locks" / "model_lock.json"
    assert lock_path.is_file(), "model_lock.json is missing from locks/"
    
    with open(lock_path, "r", encoding="utf-8") as f:
        lock = json.load(f)
        
    # Check that required keys are present
    assert "detector_onnx_sha256" in lock
    assert "coco_onnx_sha256" in lock
    assert "lprnet_onnx_sha256" in lock
    assert "crnn_onnx_sha256" in lock
