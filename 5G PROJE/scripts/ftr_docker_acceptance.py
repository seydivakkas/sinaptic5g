#!/usr/bin/env python3
"""
FTR Docker Build + Run Acceptance Test
=======================================
Docker daemon açık olduğunda çalıştırın:
    python scripts/ftr_docker_acceptance.py

Adımlar:
  1. Docker build (5G PROJE/ dizininden)
  2. Image boyutu kontrolü (< 8 GB)
  3. Docker run - normal video (results.json üretilmeli)
  4. JSON schema dogrulamasi
  5. Etiket kontrat dogrulamasi
  6. Çalışma süresi kontrolü (< 600 saniye)
  7. Network isolation testi (--network none ile çalişmali)
  8. Bozuk video dayanıklılık testi (exit code 1 olmali, container çökmemeli)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
IMAGE_NAME = "teknofest/sinaptic5g:latest"
MAX_IMAGE_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB
MAX_RUNTIME_SECONDS = 600  # 10 dakika

PASS = "[OK]  "
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def check(name, ok, note=""):
    status = PASS if ok else FAIL
    msg = f"{status} {name}" + (f" -- {note}" if note else "")
    results.append((ok, msg))
    print(msg)
    return ok

def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)

# ── 1. Docker Build ────────────────────────────────────────────────────────────
print(f"\n=== 1. Docker Build ===")
print(f"  Dizin: {PROJECT_ROOT}")
print(f"  Imaj: {IMAGE_NAME}")

t0 = time.time()
result = run(
    ["docker", "build", "-t", IMAGE_NAME, "."],
    cwd=str(PROJECT_ROOT)
)
build_time = time.time() - t0

if not check("Docker build basarili", result.returncode == 0,
             f"{build_time:.1f}s" if result.returncode == 0 else result.stderr[-500:]):
    print("\n[HATA] Build basarisiz, devam edilemiyor.")
    sys.exit(1)

# ── 2. Image Boyutu ────────────────────────────────────────────────────────────
print(f"\n=== 2. Docker Image Boyutu ===")
r = run(["docker", "image", "inspect", IMAGE_NAME, "--format", "{{.Size}}"])
if r.returncode == 0:
    size_bytes = int(r.stdout.strip())
    size_gb = size_bytes / (1024**3)
    check(f"Image boyutu {size_gb:.2f} GB < 8 GB", size_bytes < MAX_IMAGE_BYTES,
          f"{size_gb:.2f} GB")
else:
    check("Image boyutu alinamadi", False, r.stderr)

# ── 3. Docker Run - Normal Video ───────────────────────────────────────────────
print(f"\n=== 3. Docker Run - Normal Video ===")
smoke_video = PROJECT_ROOT / "tests/smoke_input/video.mp4"
if not smoke_video.is_file():
    print(f"{WARN} Smoke video bulunamadi: {smoke_video}")
    print(f"{WARN} tests/smoke_input/video.mp4 gerekiyor - atlanıyor")
else:
    with tempfile.TemporaryDirectory() as output_dir:
        t1 = time.time()
        r = run([
            "docker", "run", "--rm",
            "--network", "none",
            "-v", f"{smoke_video}:/app/data/input/video.mp4:ro",
            "-v", f"{output_dir}:/app/data/output:rw",
            IMAGE_NAME
        ])
        runtime = time.time() - t1

        check("Container exit code 0", r.returncode == 0,
              f"exit={r.returncode}" if r.returncode != 0 else f"{runtime:.1f}s")
        check(f"Calisma suresi {runtime:.1f}s < 600s", runtime < MAX_RUNTIME_SECONDS,
              f"{runtime:.1f}s")

        results_path = Path(output_dir) / "results.json"
        if check("results.json uretildi", results_path.is_file()):
            # ── 4. JSON Schema ─────────────────────────────────────────────────
            print(f"\n=== 4. JSON Schema Dogrulamasi ===")
            try:
                from jsonschema import Draft202012Validator
                schema = json.loads(
                    (PROJECT_ROOT / "schemas/results.schema.json").read_text(encoding="utf-8")
                )
                data = json.loads(results_path.read_text(encoding="utf-8"))
                Draft202012Validator(schema).validate(data)
                check("JSON schema gecti", True)
            except Exception as exc:
                check("JSON schema gecti", False, str(exc)[:200])

            # ── 5. Etiket Kontrat ──────────────────────────────────────────────
            print(f"\n=== 5. Etiket Kontrat Dogrulamasi ===")
            VALID_LABELS = {
                "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
                "telefonla_konusma", "slalom", "etrafa_bakinma",
                "emniyet_kemeri_ihlali", "teknocan", "bilgisayar",
                "arka_koltuk_1", "arka_koltuk_2", "on_koltuk"
            }
            invalid = [
                d["etiket"] for d in data.get("tespitler", [])
                if d.get("etiket") not in VALID_LABELS
            ]
            check("Tum etiketler FTR whitelist'te", len(invalid) == 0,
                  f"gecersiz: {invalid}" if invalid else "")

            import re
            raw = results_path.read_text(encoding="utf-8")
            tc = re.findall(r"[cCgGiIsSlLuUoObBrRtTyYkK]", raw)  # simplified
            # Turkce ASCII check
            turkish_chars = re.findall(r"[çğışöüÇĞIŞÖÜ]", raw)
            check("Turkce karakter yok (ASCII-safe)", len(turkish_chars) == 0,
                  f"bulunan: {set(turkish_chars)}" if turkish_chars else "")

            vid_id = data.get("video_id", "")
            check("video_id .mp4 ile bitiyor", vid_id.endswith(".mp4"), vid_id)

# ── 6. Bozuk Video Dayaniklilik Testi ──────────────────────────────────────────
print(f"\n=== 6. Bozuk Video Dayaniklilik Testi ===")
with tempfile.TemporaryDirectory() as broken_out:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as broken_f:
        broken_f.write(b"NOT_A_REAL_VIDEO_FILE_CORRUPTED")
        broken_path = broken_f.name
    try:
        r = run([
            "docker", "run", "--rm",
            "--network", "none",
            "-v", f"{broken_path}:/app/data/input/video.mp4:ro",
            "-v", f"{broken_out}:/app/data/output:rw",
            IMAGE_NAME
        ])
        check("Bozuk video -- container cakmadi (exit 1)", r.returncode == 1,
              f"exit={r.returncode}")
        no_output = not (Path(broken_out) / "results.json").is_file()
        check("Bozuk video -- results.json yazilmadi", no_output)
    finally:
        os.unlink(broken_path)

# ── Ozet ───────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for ok, _ in results if ok)
total  = len(results)
failed = [msg for ok, msg in results if not ok]
print(f"SONUC: {passed}/{total} kontrol gecti")
if failed:
    print("\nEksik/Hatali:")
    for m in failed:
        print(f"  {m}")
    sys.exit(1)
else:
    print("DOCKER ACCEPTANCE: TUM KONTROLLER BASARILI")
    sys.exit(0)
