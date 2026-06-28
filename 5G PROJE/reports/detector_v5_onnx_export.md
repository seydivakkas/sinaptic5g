# SİNAPTİC5G — detector_v5 ONNX Dışa Aktarım ve Doğrulama Raporu

> **Tarih:** 2026-06-25
> **Modül:** Phase 7 & 8 — ONNX Dışa Aktarım ve Bütünlük Kilidi

## 1. Dışa Aktarım Detayları

* **PyTorch Ağırlıkları:** `models/candidates/detector_v5/best.pt` (Size: 49.63 MB)
* **ONNX Dosya Yolu:** `models/candidates/detector_v5/detector_v5.onnx` (Size: 49.47 MB)
* **PyTorch SHA-256:** `69f579ad3e429542b64ecdeafbe7bd027d9ed8309d84be8f2775d00b7f484778`
* **ONNX SHA-256:** `da1840434b6a13eae5205ff5ce7e41c60edc52d93a11c0d422fdaa86a966d5b9`

## 2. Çıkarım Doğrulama (Inference Smoke Test)

* **Test Görüntüsü:** `cigarette_smokers_v5__010ab1e14a14.jpg`
* **Orijinal Boyutlar:** 640x640
* **Çıktı Tensor Şekli (Shape):** (1, 13, 8400)
* **Ortalama CPU Gecikmesi:** 118.68 ms

## 3. Bütünlük Kilidi Kontrolü

Ağırlıklar ve ONNX dosyası için SHA-256 bütünlük kontrol kilitleri `models/candidates/detector_v5/model_lock_v5.json` dosyasına yazılmıştır. Dosya bütünlüğü başarıyla doğrulanmıştır. Üretim kilidi (`models/detector.onnx`) korunmuş, v5 modeli aday kategorisinde güvenli bir şekilde dondurulmuştur. ✅

---
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
