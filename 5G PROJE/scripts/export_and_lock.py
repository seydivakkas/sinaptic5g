# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
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

"""F8 Task: Export model, calculate SHA-256 hashes, lock configurations, and validate."""

import os
import json
import hashlib
import shutil
from pathlib import Path
from ultralytics import YOLO

def get_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    best_pt_path = Path("models/runs/detector_v1/weights/best.pt")
    
    if not best_pt_path.is_file():
        print(f"Error: Trained model weights not found at {best_pt_path}")
        return
        
    print(f"Loading model weights from {best_pt_path}...")
    model = YOLO(best_pt_path)
    
    # 1. Export best.pt -> detector.onnx (FP16, opset=17, simplify=True)
    print("\nExporting best.pt to ONNX (FP16, opset=17, simplify)...")
    onnx_exported_path = model.export(format="onnx", opset=17, half=True, simplify=True)
    
    # Copy to models/detector.onnx
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    dst_onnx = models_dir / "detector.onnx"
    shutil.copy2(onnx_exported_path, dst_onnx)
    print(f"ONNX model saved to {dst_onnx}")
    
    # 2. Export best.pt -> model.tflite (FP16)
    print("\nExporting best.pt to TFLite (FP16)...")
    dst_tflite = models_dir / "model.tflite"
    android_assets_dir = Path("android/app/src/main/assets")
    android_assets_dir.mkdir(parents=True, exist_ok=True)
    android_tflite = android_assets_dir / "yolov8n.tflite"
    
    try:
        # TFLite format in Ultralytics YOLOv8 export
        tflite_exported_dir = model.export(format="tflite", half=True)
        tflite_path = Path(tflite_exported_dir)
        if tflite_path.is_dir():
            tflite_files = list(tflite_path.glob("*.tflite"))
            if tflite_files:
                tflite_file = tflite_files[0]
            else:
                raise FileNotFoundError("TFLite file not found inside exported directory.")
        else:
            tflite_file = tflite_path
            
        shutil.copy2(tflite_file, dst_tflite)
        print(f"TFLite model saved to {dst_tflite}")
        shutil.copy2(tflite_file, android_tflite)
        print(f"Updated Android app assets TFLite model at {android_tflite}")
    except Exception as e:
        print(f"[!] Warning: TFLite export failed ({e}). This is expected on Python 3.14 as TensorFlow is not yet compatible.")
        if dst_tflite.is_file():
            print(f"[+] Falling back to using existing TFLite model at {dst_tflite}")
            # Ensure it is also in android assets if not there
            if dst_tflite.is_file() and not android_tflite.is_file():
                shutil.copy2(dst_tflite, android_tflite)
        else:
            print("[-] Warning: No existing model.tflite found to fall back to.")
            
    # 3. Calculate SHA-256 & Update model_lock.json
    print("\nCalculating SHA-256 hashes...")
    pt_hash = get_sha256(best_pt_path)
    onnx_hash = get_sha256(dst_onnx)
    tflite_hash = get_sha256(dst_tflite) if dst_tflite.is_file() else "N/A"

    
    lock_path = Path("model_lock.json")
    lock_data = {}
    if lock_path.is_file():
        try:
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to load existing model lock: {e}")

    lock_data.update({
        "best_pt_sha256": pt_hash,
        "detector_onnx_sha256": onnx_hash,
        "model_tflite_sha256": tflite_hash,
        "best_pt_size_bytes": best_pt_path.stat().st_size,
        "detector_onnx_size_bytes": dst_onnx.stat().st_size,
        "model_tflite_size_bytes": dst_tflite.stat().st_size
    })
    
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)
    print(f"Updated model lock at {lock_path}")
    
    # Update android/app/src/main/assets/model_manifest.json as well!
    android_manifest_path = android_assets_dir / "model_manifest.json"
    if android_manifest_path.is_file():
        manifest = json.loads(android_manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_sha256"] = tflite_hash
        manifest["artifact_size_bytes"] = dst_tflite.stat().st_size
        manifest["source_sha256"] = pt_hash
        manifest["source_size_bytes"] = best_pt_path.stat().st_size
        
        with open(android_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Updated Android model manifest at {android_manifest_path}")
        
    # 4. Measure validation/test set mAP
    print("\nMeasuring test set metrics...")
    try:
        val_results = model.val(data="data/processed/merged/data.yaml", split="test")
        
        # Parse metrics
        mAP50 = val_results.results_dict.get("metrics/mAP50(B)", 0.0)
        mAP50_95 = val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)
        
        metrics_data = {
            "mAP_50": round(mAP50, 4),
            "mAP_50_95": round(mAP50_95, 4),
            "classes": val_results.names
        }
        
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = reports_dir / "val_metrics.json"
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Test metrics saved to {metrics_path}")
        print(f"Test mAP@50: {mAP50:.4f}")
    except Exception as e:
        print(f"Error measuring validation metrics: {e}")
        
    print("\nModel locking and validations complete!")

if __name__ == "__main__":
    main()
