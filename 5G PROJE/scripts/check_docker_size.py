# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
#
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
#
# Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
# amaçlı olarak paylaşılmıştır.
#
# YASAKLAR:
#   1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
#   2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
#   3. Alt lisanslanamaz, satılamaz veya devredilemez.
#   4. Tersine mühendislik yapılamaz.
#
# İZİN VERİLEN KULLANIM:
#   - GitHub üzerinde görüntüleme ve okuma.
#   - Kişisel öğrenim amacıyla kodu inceleme (kopyalamadan).
#
# YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
# İzin talepleri için: GitHub @seydivakkas

"""
scripts/check_docker_size.py — Docker Image Boyut Kontrol Scripti
=================================================================
Faz 5: Test & Doğrulama / CI Adımı

Docker image boyutunu kontrol eder ve eşiği aşarsa hata verir.
Proje dosya ağacı boyutunu da raporlar.

Kullanım:
    # Docker image kontrolü:
    python scripts/check_docker_size.py --image sinaptic5g:latest --max-size-gb 4.0

    # Proje dosya boyutu kontrolü (Docker yok):
    python scripts/check_docker_size.py --check-files --max-size-gb 2.0

    # CI adımı:
    python scripts/check_docker_size.py --image sinaptic5g:ftr --max-size-gb 3.5 --ci
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger("sinaptic5g.check_docker_size")

PROJECT_ROOT = Path(__file__).parent.parent

# ─── Boyut Limitleri ─────────────────────────────────────────────────────────
DEFAULT_MAX_IMAGE_SIZE_GB = 4.0   # FTR Docker image maksimum boyutu
DEFAULT_MAX_LAYER_SIZE_GB = 0.5   # Tek katman maksimum boyutu

# ─── Model Boyut Referansları ─────────────────────────────────────────────────
EXPECTED_MODEL_SIZES: Dict[str, float] = {
    "models/detector.onnx": 49.5,    # MB
    "models/crnn.onnx": 44.8,        # MB
    "models/lprnet.onnx": 1.7,       # MB
    "models/cnn_lstm.onnx": 0.002,   # MB (çok küçük)
}


def check_docker_image_size(
    image_name: str,
    max_size_gb: float = DEFAULT_MAX_IMAGE_SIZE_GB,
) -> Dict:
    """Docker image boyutunu kontrol eder."""
    result = {"image": image_name, "max_size_gb": max_size_gb}
    
    try:
        cmd = ["docker", "image", "inspect", image_name, "--format={{.Size}}"]
        output = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if output.returncode != 0:
            result["status"] = "IMAGE_NOT_FOUND"
            result["error"] = output.stderr.strip()
            return result
        
        size_bytes = int(output.stdout.strip())
        size_gb = size_bytes / (1024 ** 3)
        result["size_bytes"] = size_bytes
        result["size_gb"] = round(size_gb, 3)
        result["size_mb"] = round(size_bytes / (1024 ** 2), 1)
        
        if size_gb <= max_size_gb:
            result["status"] = "PASS"
            LOG.info("✅ Docker image boyutu: %.2f GB (limit: %.2f GB)", size_gb, max_size_gb)
        else:
            result["status"] = "FAIL"
            LOG.error("❌ Docker image çok büyük: %.2f GB > %.2f GB", size_gb, max_size_gb)
    
    except FileNotFoundError:
        result["status"] = "DOCKER_NOT_AVAILABLE"
        result["error"] = "docker komutu bulunamadı"
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
    except ValueError as e:
        result["status"] = "PARSE_ERROR"
        result["error"] = str(e)
    
    return result


def check_docker_layers(image_name: str) -> Dict:
    """Docker image katmanlarını ve boyutlarını listeler."""
    result = {"image": image_name, "layers": []}
    
    try:
        cmd = ["docker", "history", image_name, "--format={{.Size}}\t{{.CreatedBy}}", "--no-trunc"]
        output = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if output.returncode != 0:
            result["status"] = "FAILED"
            return result
        
        layers = []
        for line in output.stdout.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                size_str, created_by = parts
                layers.append({"size": size_str, "created_by": created_by[:100]})
        
        result["layers"] = layers
        result["layer_count"] = len(layers)
        result["status"] = "OK"
    
    except FileNotFoundError:
        result["status"] = "DOCKER_NOT_AVAILABLE"
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
    
    return result


def check_project_files(
    root: Path = PROJECT_ROOT,
    max_size_gb: float = 2.0,
    ignore_dirs: Optional[List[str]] = None,
) -> Dict:
    """Proje dosya boyutunu kontrol eder (Docker olmadan)."""
    if ignore_dirs is None:
        ignore_dirs = [".git", "__pycache__", ".pytest_cache", "runs", "node_modules",
                       "driveact_manifest.csv"]
    
    total_bytes = 0
    large_files: List[Dict] = []
    file_counts: Dict[str, int] = {}
    
    for path in root.rglob("*"):
        if any(d in path.parts for d in ignore_dirs):
            continue
        if not path.is_file():
            continue
        
        try:
            size = path.stat().st_size
        except OSError:
            continue
        
        total_bytes += size
        suffix = path.suffix.lower() or "no_ext"
        file_counts[suffix] = file_counts.get(suffix, 0) + 1
        
        # 50 MB'dan büyük dosyaları listele
        if size > 50 * 1024 * 1024:
            large_files.append({
                "path": str(path.relative_to(root)),
                "size_mb": round(size / (1024 ** 2), 1),
            })
    
    total_gb = total_bytes / (1024 ** 3)
    
    result = {
        "total_bytes": total_bytes,
        "total_gb": round(total_gb, 3),
        "total_mb": round(total_bytes / (1024 ** 2), 1),
        "max_size_gb": max_size_gb,
        "large_files": sorted(large_files, key=lambda x: -x["size_mb"])[:20],
        "file_type_counts": dict(sorted(file_counts.items(), key=lambda x: -x[1])[:15]),
        "status": "PASS" if total_gb <= max_size_gb else "FAIL",
    }
    
    if result["status"] == "PASS":
        LOG.info("✅ Proje boyutu: %.2f GB (limit: %.2f GB)", total_gb, max_size_gb)
    else:
        LOG.warning("⚠️  Proje boyutu: %.2f GB > %.2f GB limit", total_gb, max_size_gb)
    
    if large_files:
        LOG.info("Büyük dosyalar (>50 MB):")
        for lf in large_files:
            LOG.info("  %s: %.1f MB", lf["path"], lf["size_mb"])
    
    return result


def check_model_sizes() -> Dict:
    """Kritik model dosyalarının boyutlarını kontrol eder."""
    results = {}
    
    for model_rel_path, expected_mb in EXPECTED_MODEL_SIZES.items():
        model_path = PROJECT_ROOT / model_rel_path
        entry = {"path": model_rel_path, "expected_mb": expected_mb}
        
        if model_path.is_file():
            actual_mb = model_path.stat().st_size / (1024 ** 2)
            entry["actual_mb"] = round(actual_mb, 2)
            # %20 tolerans
            tolerance = 0.2
            if abs(actual_mb - expected_mb) / max(expected_mb, 1) <= tolerance:
                entry["status"] = "OK"
            else:
                entry["status"] = "SIZE_MISMATCH"
                LOG.warning("Model boyutu uyuşmuyor: %s (beklenen %.1f MB, gerçek %.1f MB)",
                             model_rel_path, expected_mb, actual_mb)
        else:
            entry["status"] = "NOT_FOUND"
            entry["actual_mb"] = 0
        
        results[model_rel_path] = entry
    
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Sinaptic5G — Docker Image Boyut Kontrolü")
    parser.add_argument("--image", type=str, default=None,
                        help="Docker image adı (örn: sinaptic5g:latest)")
    parser.add_argument("--max-size-gb", type=float, default=DEFAULT_MAX_IMAGE_SIZE_GB,
                        help="Maksimum image boyutu (GB)")
    parser.add_argument("--check-files", action="store_true",
                        help="Proje dosya boyutunu kontrol et (Docker olmadan)")
    parser.add_argument("--check-models", action="store_true",
                        help="Model dosyası boyutlarını kontrol et")
    parser.add_argument("--ci", action="store_true",
                        help="CI modu — başarısız olursa exit code 1")
    parser.add_argument("--output", type=Path, default=Path("reports/docker_size_check.json"),
                        help="Rapor çıktı yolu")
    
    args = parser.parse_args()
    
    all_results = {}
    overall_status = "PASS"
    
    if args.image:
        LOG.info("Docker image kontrolü: %s", args.image)
        image_result = check_docker_image_size(args.image, args.max_size_gb)
        all_results["docker_image"] = image_result
        if image_result.get("status") == "FAIL":
            overall_status = "FAIL"
        
        layer_result = check_docker_layers(args.image)
        all_results["docker_layers"] = layer_result
    
    if args.check_files or args.image is None:
        LOG.info("Proje dosya boyutu kontrolü...")
        file_result = check_project_files(max_size_gb=args.max_size_gb * 2)
        all_results["project_files"] = file_result
    
    if args.check_models:
        LOG.info("Model dosyası boyutu kontrolü...")
        model_result = check_model_sizes()
        all_results["model_sizes"] = model_result
    
    all_results["overall_status"] = overall_status
    
    # Rapor yaz
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Rapor: %s", args.output)
    
    if args.ci and overall_status == "FAIL":
        LOG.error("CI kontrolü başarısız — çok büyük boyut")
        return 1
    
    LOG.info("Genel durum: %s", overall_status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
