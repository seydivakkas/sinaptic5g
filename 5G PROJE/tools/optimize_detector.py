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

import os
import sys
import json
import hashlib
from pathlib import Path
import onnxruntime as ort

def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_model(path: Path) -> bool:
    try:
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # Try loading the session to verify integrity
        ort.InferenceSession(str(path), sess_opt, providers=["CPUExecutionProvider"])
        return True
    except Exception as e:
        print(f"[!] Model verification failed for {path.name}: {e}")
        return False

def main():
    model_path = Path("models/detector.onnx")
    output_path = Path("models/detector_optimized.onnx")
    model_lock_path = Path("model_lock.json")
    
    if not model_path.is_file():
        model_path = Path("5G PROJE/models/detector.onnx")
        output_path = Path("5G PROJE/models/detector_optimized.onnx")
        
    if not model_path.is_file():
        print(f"[-] Error: detector.onnx not found at {model_path}")
        sys.exit(1)
        
    print(f"[*] Optimizing ONNX graph for {model_path} -> {output_path}...")
    try:
        # Configure SessionOptions to save the optimized model
        sess_opt = ort.SessionOptions()
        sess_opt.optimized_model_filepath = str(output_path)
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load session to trigger and save graph optimizations
        ort.InferenceSession(str(model_path), sess_opt, providers=["CPUExecutionProvider"])
        print("[+] Graph optimization and serialization completed successfully.")
        
        # Verify the optimized model
        if verify_model(output_path):
            print("[+] Optimized model verified successfully.")
        else:
            print("[!] Optimized model verification failed. Reverting to original model copy...")
            import shutil
            output_path.unlink(missing_ok=True)
            shutil.copy2(model_path, output_path)
            print("[+] Original model copied to detector_optimized.onnx as fallback.")
            
        # Calculate hash and update model_lock.json
        h = compute_sha256(output_path)
        print(f"[+] detector_optimized.onnx SHA-256: {h}")
        
        if model_lock_path.is_file():
            lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
            # Remove old key if exists
            lock.pop("detector_int8_onnx_sha256", None)
            lock["detector_optimized_onnx_sha256"] = h
            model_lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
            print("[+] Successfully updated model_lock.json: detector_optimized_onnx_sha256")
            
    except Exception as e:
        print(f"[-] Optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
