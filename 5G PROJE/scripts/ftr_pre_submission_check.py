"""
FTR Teslim Öncesi Kabul Testi Suite
=====================================
Docker daemon kapalıyken lokal Python ortamında çalıştırılabilir.
Kapsam:
  1. Model lock SHA-256 dogrulamasi
  2. results.json şema + etiket kontrat dogrulamasi (mevcut smoke çıktısı)
  3. ftr_main.py statik yol kontrol dogrulamasi
  4. Dockerfile içerik dogrulamasi
  5. requirements-ftr.txt mediapipe kontrolü
"""
import hashlib
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            d.update(chunk)
    return d.hexdigest()

PASS = "[OK]  "
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def check(name, ok, note=""):
    status = PASS if ok else FAIL
    msg = f"{status} {name}" + (f" — {note}" if note else "")
    results.append((ok, msg))
    print(msg)

# ── 1. Model Lock ──────────────────────────────────────────────────────────────
print("\n=== 1. Model Lock SHA-256 Dogrulamasi ===")
lock_path = PROJECT_ROOT / "model_lock.json"
lock_ok = lock_path.is_file()
check("model_lock.json mevcut", lock_ok)

if lock_ok:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    model_checks = {
        PROJECT_ROOT / "models/detector_optimized.onnx": lock.get("detector_onnx_sha256") or lock.get("detector_optimized_onnx_sha256"),
        PROJECT_ROOT / "yolov8n.onnx":                  lock.get("coco_onnx_sha256"),
        PROJECT_ROOT / "models/lprnet.onnx":             lock.get("lprnet_onnx_sha256"),
        PROJECT_ROOT / "models/crnn.onnx":               lock.get("crnn_onnx_sha256"),
    }
    for path, expected in model_checks.items():
        if not path.is_file():
            check(f"Model mevcut: {path.name}", False, "dosya yok")
            continue
        actual = sha256(path)
        ok = actual.lower() == str(expected).lower()
        check(f"SHA-256: {path.name}", ok,
              f"{actual[:16]}..." if ok else f"beklenen={str(expected)[:16]} gerçek={actual[:16]}")

# ── 2. JSON Şema + Etiket Kontrat ─────────────────────────────────────────────
print("\n=== 2. JSON Schema + Etiket Kontrat Dogrulamasi ===")
smoke_result = PROJECT_ROOT / "tests/final_smoke_output/results.json"

if smoke_result.is_file():
    try:
        from jsonschema import Draft202012Validator
        schema_path = PROJECT_ROOT / "schemas/results.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        data = json.loads(smoke_result.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(data)
        check("JSON schema dogrulamasi", True, smoke_result.name)

        # Zorunlu alanlar
        for field in ("video_id", "arac_bilgisi", "tespitler"):
            check(f"Zorunlu alan: {field}", field in data)

        # Etiket whitelist
        VALID_LABELS = {
            "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
            "telefonla_konusma", "slalom", "etrafa_bakinma",
            "emniyet_kemeri_ihlali", "teknocan", "bilgisayar",
            "arka_koltuk_1", "arka_koltuk_2", "on_koltuk"
        }
        invalid_labels = []
        for det in data.get("tespitler", []):
            lbl = det.get("etiket", "")
            if lbl not in VALID_LABELS:
                invalid_labels.append(lbl)
        check("Tüm etiketler FTR whitelist'te", len(invalid_labels) == 0,
              f"geçersiz: {invalid_labels}" if invalid_labels else "")

        # Türkçe karakter kontrolü
        raw = smoke_result.read_text(encoding="utf-8")
        turkish_chars = re.findall(r"[çğışöüÇĞIŞÖÜ]", raw)
        check("Türkçe karakter yok", len(turkish_chars) == 0,
              f"bulunan: {set(turkish_chars)}" if turkish_chars else "ASCII-safe")

        # confidence_score [0,1]
        scores_ok = True
        if not (0 <= data["arac_bilgisi"]["confidence_score"] <= 1):
            scores_ok = False
        for det in data.get("tespitler", []):
            if not (0 <= det.get("confidence_score", -1) <= 1):
                scores_ok = False
        check("Tüm confidence_score [0,1] aralıgında", scores_ok)

    except Exception as exc:
        check("JSON schema dogrulamasi", False, str(exc))
else:
    print(f"{WARN} Smoke çıktısı bulunamadı: {smoke_result} (docker run sonrası tekrar çalıstırın)")

# ── 3. ftr_main.py Statik Yol Kontrolü ────────────────────────────────────────
print("\n=== 3. ftr_main.py Statik Yol + FTR Madde 5 Dogrulamasi ===")
ftr_main = PROJECT_ROOT / "ftr_core/ftr_main.py"
txt = ftr_main.read_text(encoding="utf-8")

banned = {
    "os.getenv(FTR_INPUT_PATH)":  r'getenv.*FTR_INPUT_PATH',
    "os.getenv(FTR_OUTPUT_PATH)": r'getenv.*FTR_OUTPUT_PATH',
    "is_file() coco fallback":    r'is_file\(\).*coco|coco.*is_file\(\)',
    "hostname/IP kontrolü":       r'socket\.gethostname|os\.environ\.get.*HOST',
}
for name, pattern in banned.items():
    found = bool(re.search(pattern, txt))
    check(f"Yasaklı pattern yok: {name}", not found, "diskalifiye riski" if found else "")

required_static = [
    "/app/data/input/video.mp4",
    "/app/data/output/results.json",
    "/app/models/coco.onnx",
    "/app/models/detector_optimized.onnx",
    "/app/model_lock.json",
]
for path in required_static:
    check(f"Statik yol mevcut: {path}", path in txt)

# ── 4. Dockerfile Kontrolü ────────────────────────────────────────────────────
print("\n=== 4. Dockerfile Içerik Dogrulamasi ===")
dockerfile = PROJECT_ROOT.parent / "Dockerfile"
if not dockerfile.is_file():
    dockerfile = PROJECT_ROOT / "docker/Dockerfile"
    if not dockerfile.is_file():
        dockerfile = PROJECT_ROOT / "Dockerfile"
df = dockerfile.read_text(encoding="utf-8")

df_checks = {
    "CUDA base image": "nvidia/cuda:12.1.0-base-ubuntu22.04",
    "WORKDIR /app":    "WORKDIR /app",
    "ENTRYPOINT":      'ENTRYPOINT ["python3", "/app/main.py"]',
    "plate_ocr.py":    "plate_ocr.py /app/plate_ocr.py",
    "driver_analyzer": "driver_analyzer.py /app/driver_analyzer.py",
    "coco.onnx target":"/app/models/coco.onnx",
    "--network none note": "network none",
}
for name, needle in df_checks.items():
    check(f"Dockerfile: {name}", needle in df)

# ── 5. requirements-ftr.txt ────────────────────────────────────────────────────
print("\n=== 5. requirements-ftr.txt Kontrolü ===")
req = (PROJECT_ROOT / "requirements-ftr.txt").read_text(encoding="utf-8")
for pkg in ("onnxruntime-gpu", "opencv-python-headless", "jsonschema", "mediapipe==0.10.14"):
    check(f"Paket mevcut: {pkg}", pkg in req)

# ── Özet ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for ok, _ in results if ok)
total  = len(results)
failed_items = [msg for ok, msg in results if not ok]
print(f"SONUC: {passed}/{total} kontrol geçti")
if failed_items:
    print("\nEksik/Hatalı:")
    for m in failed_items:
        print(" ", m)
else:
    print("TUM KONTROLLER BASARILI -- FTR_READY_WITH_FIXES => FTR_READY")
