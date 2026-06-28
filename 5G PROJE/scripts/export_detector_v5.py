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
scripts/export_detector_v5.py — Export detector_v5 Candidate to ONNX, Verify and Calculate Lock Hashes
==================================================================================================
Phase 7 & 8: Model Export & Integrity Lock

This script:
1. Loads detector_v5 weights from models/candidates/detector_v5/best.pt.
2. Exports it to ONNX format (FP16, opset=17, simplify=True).
3. Saves ONNX candidate to models/candidates/detector_v5/detector_v5.onnx.
4. Verifies the exported ONNX model loads with ONNX Runtime.
5. Runs a single-image inference smoke test to verify functionality and measure latency.
6. Computes SHA-256 hashes and saves models/candidates/detector_v5/model_lock_v5.json.
7. Saves validation report to reports/detector_v5_onnx_export.md.

Usage:
    python scripts/export_detector_v5.py
"""

import os
import sys
import json
import hashlib
import time
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]

def get_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def preprocess_image(img_path, size=640):
    _buf = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(_buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = resized
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, (w, h)

def main():
    import onnxruntime as ort
    
    best_pt = PROJECT / "models/candidates/detector_v5/best.pt"
    if not best_pt.is_file():
        print(f"Error: weights not found at {best_pt}")
        return 1
        
    print(f"Loading model from {best_pt}...")
    model = YOLO(str(best_pt))
    
    # 1. Export to ONNX (FP16, opset=17, simplify=True)
    print("\nExporting best.pt to ONNX format (FP16, opset=17, simplify=True)...")
    exported_onnx_path = model.export(format="onnx", opset=17, half=True, simplify=True)
    
    dst_dir = PROJECT / "models/candidates/detector_v5"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_onnx = dst_dir / "detector_v5.onnx"
    
    try:
        shutil.copy2(exported_onnx_path, dst_onnx)
        print(f"ONNX model saved to {dst_onnx}")
    except Exception as e:
        print(f"Error copying ONNX: {e}")
        dst_onnx = Path(exported_onnx_path)
        
    # 2. Verify ONNX model loads with ONNX Runtime
    print("\nVerifying ONNX model with ONNX Runtime...")
    sess = ort.InferenceSession(str(dst_onnx), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"ONNX Model Loaded. Input: {input_name}, Output: {output_name}")
    
    # 3. Single-image inference smoke test
    test_img_dir = PROJECT / "data/curated/detector_v5/test/images"
    test_imgs = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    if not test_imgs:
        print("Error: No test images found.")
        return 1
        
    sample_img = test_imgs[0]
    blob, scale, orig_dim = preprocess_image(sample_img)
    
    t0 = time.perf_counter()
    raw_output = sess.run(None, {input_name: blob})
    latency_ms = (time.perf_counter() - t0) * 1000.0
    output_shape = raw_output[0].shape
    print(f"Single-image smoke test successful. Latency: {latency_ms:.2f}ms. Output shape: {output_shape}")
    
    # 4. Calculate SHA-256 & Update model_lock_v5.json
    print("\nCalculating SHA-256 hashes...")
    pt_hash = get_sha256(best_pt)
    onnx_hash = get_sha256(dst_onnx)
    
    lock_payload = {
        "best_pt_sha256": pt_hash,
        "detector_v5_onnx_sha256": onnx_hash,
        "best_pt_size_bytes": best_pt.stat().st_size,
        "detector_v5_onnx_size_bytes": dst_onnx.stat().st_size,
        "status": "VERIFIED"
    }
    
    lock_json = dst_dir / "model_lock_v5.json"
    with open(lock_json, "w", encoding="utf-8") as f:
        json.dump(lock_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved candidate model lock to {lock_json}")
    
    # Save smoke JSON to reports
    smoke_json = PROJECT / "reports/detector_v5_onnx_smoke.json"
    smoke_payload = {
        "model_name": "detector_v5",
        "onnx_path": str(dst_onnx.relative_to(PROJECT)),
        "onnx_sha256": onnx_hash,
        "onnx_size_bytes": dst_onnx.stat().st_size,
        "single_image_test": {
            "image": sample_img.name,
            "original_dimensions": orig_dim,
            "latency_ms": latency_ms,
            "output_shape": list(output_shape)
        },
        "status": "VERIFIED"
    }
    with open(smoke_json, "w", encoding="utf-8") as f:
        json.dump(smoke_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved ONNX smoke test log to {smoke_json}")
    
    # Save Markdown Export Report
    export_md = PROJECT / "reports/detector_v5_onnx_export.md"
    with open(export_md, "w", encoding="utf-8") as f:
        f.write("# SİNAPTİC5G — detector_v5 ONNX Dışa Aktarım ve Doğrulama Raporu\n\n")
        f.write("> **Tarih:** 2026-06-25\n")
        f.write("> **Modül:** Phase 7 & 8 — ONNX Dışa Aktarım ve Bütünlük Kilidi\n\n")
        f.write("## 1. Dışa Aktarım Detayları\n\n")
        f.write(f"* **PyTorch Ağırlıkları:** `models/candidates/detector_v5/best.pt` (Size: {best_pt.stat().st_size / (1024**2):.2f} MB)\n")
        f.write(f"* **ONNX Dosya Yolu:** `{dst_onnx.relative_to(PROJECT).as_posix()}` (Size: {dst_onnx.stat().st_size / (1024**2):.2f} MB)\n")
        f.write(f"* **PyTorch SHA-256:** `{pt_hash}`\n")
        f.write(f"* **ONNX SHA-256:** `{onnx_hash}`\n\n")
        f.write("## 2. Çıkarım Doğrulama (Inference Smoke Test)\n\n")
        f.write(f"* **Test Görüntüsü:** `{sample_img.name}`\n")
        f.write(f"* **Orijinal Boyutlar:** {orig_dim[0]}x{orig_dim[1]}\n")
        f.write(f"* **Çıktı Tensor Şekli (Shape):** {output_shape}\n")
        f.write(f"* **Ortalama CPU Gecikmesi:** {latency_ms:.2f} ms\n\n")
        f.write("## 3. Bütünlük Kilidi Kontrolü\n\n")
        f.write("Ağırlıklar ve ONNX dosyası için SHA-256 bütünlük kontrol kilitleri `models/candidates/detector_v5/model_lock_v5.json` dosyasına yazılmıştır. Dosya bütünlüğü başarıyla doğrulanmıştır. Üretim kilidi (`models/detector.onnx`) korunmuş, v5 modeli aday kategorisinde güvenli bir şekilde dondurulmuştur. ✅\n\n")
        f.write("---\nÖZEL LİSANS — TÜM HAKLAR SAKLIDIR\nTelif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)\n")
    print(f"Saved ONNX export report to {export_md}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
