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
import json
import hashlib
from pathlib import Path

def get_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    
    # Hedef model dosyaları
    models = {
        "detector_onnx": models_dir / "detector_optimized.onnx",
        "coco_onnx": root / "yolov8n.onnx", # YOLOv8n.onnx root'ta bulunuyor
        "lprnet_onnx": models_dir / "lprnet.onnx",
        "crnn_onnx": models_dir / "crnn.onnx",
        "cnn_lstm_onnx": models_dir / "cnn_lstm.onnx",
    }
    
    lock_data = {}
    for key, path in models.items():
        if path.exists():
            sha256 = get_sha256(path)
            size = path.stat().st_size
            lock_data[f"{key}_sha256"] = sha256
            lock_data[f"{key}_size_bytes"] = size
            print(f"[OK] {key} hesaplandı: {sha256} ({size} bayt)")
        else:
            print(f"[WARNING] {path} bulunamadı, kilitlenemiyor!")
            
    # Kaydet
    output_path = root / "model_lock.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Model kilidi başarıyla yazıldı: {output_path}")

if __name__ == "__main__":
    main()
