"""Acceptance scan: one TFLite, no ONNX, no configured secret value."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APK = ROOT / "android/app/build/outputs/apk/debug/app-debug.apk"
MANIFEST = ROOT / "android/app/src/main/assets/model_manifest.json"


def configured_secrets() -> list[bytes]:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return []
    values = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if ("SECRET" in name or "TOKEN" in name) and len(value) >= 12:
            values.append(value.encode("utf-8"))
    return values


def main() -> None:
    apk = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else APK
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not apk.is_file():
        raise SystemExit(f"APK missing: {apk}")
    with zipfile.ZipFile(apk) as archive:
        names = archive.namelist()
        tflite = [name for name in names if name.lower().endswith(".tflite")]
        onnx = [name for name in names if name.lower().endswith(".onnx")]
        if len(tflite) != 1:
            raise SystemExit(f"expected one TFLite, found {len(tflite)}")
        if onnx:
            raise SystemExit("ONNX must not be packaged in the APK")
        model_bytes = archive.read(tflite[0])
        if hashlib.sha256(model_bytes).hexdigest() != manifest["artifact_sha256"]:
            raise SystemExit("packaged TFLite hash does not match manifest")
        secrets = configured_secrets()
        for name in names:
            payload = archive.read(name)
            if any(secret in payload for secret in secrets):
                raise SystemExit("a configured secret value was found in the APK")
    print(json.dumps({
        "apk": str(apk),
        "apk_size_bytes": apk.stat().st_size,
        "tflite_count": 1,
        "onnx_count": 0,
        "configured_secret_matches": 0,
        "model_sha256": manifest["artifact_sha256"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

