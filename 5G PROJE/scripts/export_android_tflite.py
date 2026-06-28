"""Export the verified COCO source model as the Android vehicle sentinel.

Run this under Python 3.12 after installing ``requirements-export.txt``.
The Ultralytics command is intentionally fixed by the canonical plan.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "yolov8n.pt"
ANDROID_ASSET = ROOT / "android/app/src/main/assets/yolov8n.tflite"
ANDROID_MANIFEST = ROOT / "android/app/src/main/assets/model_manifest.json"
ANDROID_LOCK = ROOT / "android/model_lock.json"
EVIDENCE_MANIFEST = ROOT / "models/android_vehicle_sentinel.manifest.json"
MODEL_CARD = ROOT / "models/MODEL_CARD_android_vehicle_sentinel.md"
EXPORT_COMMAND = "yolo export model=yolov8n.pt format=tflite half=True"
EXPECTED_ULTRALYTICS = "8.4.41"

DOMAIN_ENUM = ["license_plate", "phone", "cabin", "laptop", "teknocan", "vehicle"]
DOMAIN_ALIASES = {
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_details(model_path: Path) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return {
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": input_detail["dtype"].__name__,
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": output_detail["dtype"].__name__,
    }


def main() -> None:
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise SystemExit("TFLite export requires the pinned Python 3.12 environment")
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
        raise SystemExit(f"Ultralytics {EXPECTED_ULTRALYTICS} required")
    if not SOURCE.is_file():
        raise SystemExit(f"source model missing: {SOURCE}")

    source_hash = sha256(SOURCE)
    source_model = YOLO(str(SOURCE))
    labels = [source_model.names[index] for index in sorted(source_model.names)]
    mapping = {
        str(index): DOMAIN_ALIASES[label]
        for index, label in enumerate(labels)
        if label in DOMAIN_ALIASES
    }

    with tempfile.TemporaryDirectory(prefix="sinaptic-tflite-export-") as directory:
        workdir = Path(directory)
        shutil.copy2(SOURCE, workdir / SOURCE.name)
        print(f"source_sha256={source_hash}")
        print(f"command={EXPORT_COMMAND}")
        subprocess.run(
            ["yolo", "export", "model=yolov8n.pt", "format=tflite", "half=True"],
            cwd=workdir,
            check=True,
        )
        produced = workdir / "yolov8n_saved_model/yolov8n_float16.tflite"
        if not produced.is_file():
            raise SystemExit("Ultralytics did not produce yolov8n_float16.tflite")
        details = tensor_details(produced)
        if details["input_shape"] != [1, 640, 640, 3] or details["output_shape"] != [1, 84, 8400]:
            raise SystemExit(f"unexpected tensor contract: {details}")
        ANDROID_ASSET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, ANDROID_ASSET)

    manifest = {
        "schema_version": 1,
        "model_asset": ANDROID_ASSET.name,
        "source_model": SOURCE.name,
        "source_sha256": source_hash,
        "source_size_bytes": SOURCE.stat().st_size,
        "artifact_sha256": sha256(ANDROID_ASSET),
        "artifact_size_bytes": ANDROID_ASSET.stat().st_size,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "ultralytics_version": ultralytics.__version__,
        "tensorflow_version": tf.__version__,
        "export_command": EXPORT_COMMAND,
        **details,
        "labels": labels,
        "domain_enum": DOMAIN_ENUM,
        "domain_mapping": mapping,
        "active_capabilities": sorted(set(mapping.values())),
        "nullable_capabilities": sorted(set(DOMAIN_ENUM) - set(mapping.values())),
        "evidence": {
            "artifact": "OLCULDU",
            "accuracy": "HEDEF",
            "device_latency": "HEDEF",
        },
    }
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ANDROID_MANIFEST.write_text(serialized, encoding="utf-8")
    EVIDENCE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_MANIFEST.write_text(serialized, encoding="utf-8")
    ANDROID_LOCK.write_text(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "model_asset", "artifact_sha256", "artifact_size_bytes",
                    "input_shape", "input_dtype", "output_shape", "output_dtype",
                )
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    MODEL_CARD.write_text(
        "# Android Vehicle Sentinel Model Card\n\n"
        "- Purpose: COCO car/bus/truck detection for local approach gating only.\n"
        "- Not a competition task model and not evidence for plate/type/color/behavior KPIs.\n"
        f"- Source SHA-256: `{source_hash}`\n"
        f"- Artifact SHA-256: `{manifest['artifact_sha256']}`\n"
        f"- Export: `{EXPORT_COMMAND}`\n"
        "- Active domain capability: `vehicle`; all other domain fields remain null.\n"
        "- Accuracy and device latency status: `HEDEF` until independent measurements exist.\n"
        "- INT8 and Zero-DCE stay blocked until calibration and paired-benefit evidence exist.\n",
        encoding="utf-8",
    )
    print(f"artifact={ANDROID_ASSET} sha256={manifest['artifact_sha256']}")


if __name__ == "__main__":
    main()
