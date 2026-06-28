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
scripts/promote_detector_v5.py — Promote detector_v5 ONNX model to Active Production
==================================================================================
Phase 8: Active Production Deployment & Model Lock

This script:
1. Copies models/candidates/detector_v5/detector_v5.onnx to models/detector.onnx.
2. Updates model_lock.json in the project root with the candidate's SHA-256 hashes.
3. Documents the deployment details.

Usage:
    python scripts/promote_detector_v5.py
"""

import os
import sys
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

def main():
    src_onnx = PROJECT / "models/candidates/detector_v5/detector_v5.onnx"
    dst_onnx = PROJECT / "models/detector.onnx"
    lock_json = PROJECT / "model_lock.json"
    candidate_lock = PROJECT / "models/candidates/detector_v5/model_lock_v5.json"
    
    if not src_onnx.is_file():
        print(f"Error: Candidate ONNX file not found at {src_onnx}")
        return 1
        
    if not candidate_lock.is_file():
        print(f"Error: Candidate model lock info not found at {candidate_lock}")
        return 1
        
    # Load candidate hashes
    with open(candidate_lock, "r", encoding="utf-8") as f:
        c_data = json.load(f)
        
    # 1. Backup old models/detector.onnx and detector_optimized.onnx
    if dst_onnx.is_file():
        backup_onnx = PROJECT / "models/detector_v3_backup.onnx"
        shutil.copy2(dst_onnx, backup_onnx)
        print(f"Backed up active production models/detector.onnx to {backup_onnx}")
        
    opt_onnx = PROJECT / "models/detector_optimized.onnx"
    if opt_onnx.is_file():
        backup_opt = PROJECT / "models/detector_optimized_v3_backup.onnx"
        shutil.copy2(opt_onnx, backup_opt)
        print(f"Backed up active optimized models/detector_optimized.onnx to {backup_opt}")
        
    # 2. Copy candidate ONNX to models/detector.onnx and models/detector_optimized.onnx
    shutil.copy2(src_onnx, dst_onnx)
    print("Promoted candidate detector_v5.onnx to active production models/detector.onnx [SUCCESS]")
    if opt_onnx.is_file():
        shutil.copy2(src_onnx, opt_onnx)
        print("Promoted candidate detector_v5.onnx to active production models/detector_optimized.onnx [SUCCESS]")
    
    # 3. Update model_lock.json in the root directory
    if lock_json.is_file():
        with open(lock_json, "r", encoding="utf-8") as f:
            lock_data = json.load(f)
    else:
        lock_data = {}
        
    # Backup old lock file
    backup_lock = PROJECT / "model_lock_v3_backup.json"
    with open(backup_lock, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)
    print(f"Backed up model_lock.json to {backup_lock}")
    
    # Update values
    lock_data.update({
        "best_pt_sha256": c_data["best_pt_sha256"],
        "detector_onnx_sha256": c_data["detector_v5_onnx_sha256"],
        "best_pt_size_bytes": c_data["best_pt_size_bytes"],
        "detector_onnx_size_bytes": c_data["detector_v5_onnx_size_bytes"],
        "detector_optimized_onnx_sha256": c_data["detector_v5_onnx_sha256"]  # Keep same as active
    })
    
    with open(lock_json, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)
    print("Updated model_lock.json with candidate v5 integrity parameters [SUCCESS]")
    
    print("\nPromotion complete! detector_v5 is now successfully in active production.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
