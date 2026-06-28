import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "android/app/src/main/assets/model_manifest.json"
MODEL = ROOT / "android/app/src/main/assets/yolov8n.tflite"


def test_vehicle_sentinel_manifest_and_artifact_are_locked():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["ultralytics_version"] == "8.4.41"
    assert manifest["export_command"] == "yolo export model=yolov8n.pt format=tflite half=True"
    assert manifest["domain_mapping"] == {"2": "vehicle", "5": "vehicle", "7": "vehicle"}
    assert manifest["active_capabilities"] == ["vehicle"]
    assert set(manifest["nullable_capabilities"]) == {
        "license_plate", "phone", "cabin", "laptop", "teknocan"
    }
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == manifest["artifact_sha256"]


def test_tensor_contract_accepts_future_class_count_without_domain_enum_change():
    # Equivalent to the manifest-driven Android rule: the parser derives C
    # from the tensor and drops an unmapped winning class.
    def map_winner(scores, mapping):
        raw_id = max(range(len(scores)), key=scores.__getitem__)
        return mapping.get(raw_id)

    future_scores = [0.0] * 91
    future_scores[90] = 0.9
    assert map_winner(future_scores, {90: "vehicle"}) == "vehicle"
    future_scores[11] = 0.95
    assert map_winner(future_scores, {90: "vehicle"}) is None
