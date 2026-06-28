# SİNAPTİC5G — Model ve Metrik Denetim Raporu

> **Sürüm:** 1.1  
> **Tarih:** 2026-06-26  
> **Yazar:** Seydi Eryılmaz (@seydivakkas)  
> **Kanıt Kaynakları:** `reports/detector_v5_test_metrics.json`, `reports/detector_v5_val_metrics.json`, `model_lock.json`, `reports/final_ftr_acceptance_report.md`

---

## 1. Aktif Üretim Modeli — detector_v5

### 1.1 Model Kimliği ve Bütünlük

| Parametre | Değer |
|-----------|-------|
| **Model Adı** | detector_v5 |
| **Temel Mimari** | YOLOv8m (Ultralytics 8.4.41) |
| **Eğitim Ortamı** | NVIDIA RTX 4070 Laptop GPU (8GB VRAM), CUDA 12.6, PyTorch 2.11.0+cu126 |
| **Eğitim Süresi** | 60 epoch |
| **PyTorch Ağırlığı** | `models/candidates/detector_v5/best.pt` (49.63 MB) |
| **ONNX Yolu** | `models/detector.onnx` ve `models/detector_optimized.onnx` |
| **ONNX Boyutu** | 49.47 MB |
| **SHA-256 (best.pt)** | `69f579ad3e429542b64ecdeafbe7bd027d9ed8309d84be8f2775d00b7f484778` |
| **SHA-256 (detector.onnx)** | `da1840434b6a13eae5205ff5ce7e41c60edc52d93a11c0d422fdaa86a966d5b9` |

### 1.2 Test Split Küresel Metrikleri

> **Kaynak:** `reports/detector_v5_test_metrics.json`

| Metrik | detector_v5 (Aktif) |
|--------|:--------------------:|
| **mAP@0.50** | **0.9473** |
| **mAP@0.50:0.95** | **0.6904** |
| **Precision** | **0.9379** |
| **Recall** | **0.9333** |
| **F1** | **0.9356** |

### 1.3 Test Split Sınıf Bazlı Metrikler

> **Kaynak:** `reports/detector_v5_test_metrics.json`

| Sınıf ID | Sınıf Adı | N (Test) | Precision | Recall | AP50 | AP50-95 |
|:--------:|-----------|:---:|:---:|:---:|:---:|:---:|
| 0 | telefonla_konusma | 250 | 0.9887 | 0.9760 | **0.9914** | 0.6938 |
| 1 | su_icme | 100 | 0.9884 | 1.0000 | **0.9950** | 0.8307 |
| 2 | arkaya_bakma | 101 | 0.9869 | 1.0000 | **0.9950** | 0.8367 |
| 3 | esneme | 102 | 0.9528 | 0.9895 | **0.9923** | 0.6264 |
| 4 | sigara_icme | 758 | 0.9130 | 0.7614 | **0.8577** | 0.5076 |
| 5 | emniyet_kemeri_ihlali | 93 | 0.9209 | 0.8280 | **0.8596** | 0.5887 |
| 6 | teknocan | 36 | 0.9066 | 0.9167 | **0.8973** | 0.6722 |
| 7 | bilgisayar | 26 | 0.7526 | 0.9615 | **0.9531** | 0.4719 |
| 8 | license_plate | 219 | 0.9505 | 0.9644 | **0.9841** | 0.7900 |

### 1.4 Doğrulama (Val) Split Metrikleri

> **Kaynak:** `reports/detector_v5_val_metrics.json`

| Metrik | Değer |
|--------|-------|
| mAP@0.50 | 0.9505 |
| mAP@0.50:0.95 | 0.6960 |
| Precision | 0.9280 |
| Recall | 0.9422 |
| F1 | 0.9350 |

---

## 2. Kabul Kapısı (Promotion Gate) Doğrulaması

> **Kaynak:** `reports/detector_v5_vs_v3_comparison.md`

| Kriter | Eşik | Elde Edilen | Durum |
|--------|:----:|:-----------:|:-----:|
| `teknocan` Recall | ≥ 0.30 | **0.9167** | ✅ GEÇTİ |
| `bilgisayar` AP50 | ≥ 0.10 | **0.9531** | ✅ GEÇTİ |
| `emniyet_kemeri_ihlali` Recall | ≥ 0.30 | **0.8280** | ✅ GEÇTİ |
| Genel mAP50 (Test) | ≥ 0.9164 | **0.9473** | ✅ GEÇTİ |

**Sonuç:** ✅ Tüm kabul kapısı kriterleri geçilmiştir. detector_v5 aktif üretim modeli olarak dondurulmuştur.

---

## 3. ONNX Dışa Aktarım Bilgileri

> **Kaynak:** `reports/detector_v5_onnx_export.md`

| Parametre | Değer |
|-----------|-------|
| **ONNX Opset** | 17 |
| **Precision** | FP16 (Half) |
| **Simplify** | Evet |
| **Giriş Tensörü** | (1, 3, 640, 640) |
| **Çıkış Tensörü** | (1, 13, 8400) |
| **CPU Gecikme (Ortalama)** | 118.68 ms/frame |

---

## 4. Model Versiyonlama Bilgisi

| Model | Epoch | mAP50 (Test) | Durum |
|-------|:-----:|:------------:|-------|
| **detector_v5** | **60** | **0.9473** | **Aktif üretim** |

---

## 5. Eğitim İlerleme Özeti (Epoch Metrikleri)

> **Kaynak:** `reports/detector_v5_training_log.md`

| Epoch | Val mAP50 | Val Precision | Val Recall |
|:-----:|:---------:|:-------------:|:----------:|
| 1 | ~0.45 | ~0.60 | ~0.50 |
| 10 | ~0.89 | ~0.88 | ~0.87 |
| 20 | ~0.92 | ~0.91 | ~0.90 |
| 30 | ~0.93 | ~0.92 | ~0.92 |
| 50 | ~0.939 | ~0.941 | ~0.925 |
| **60 (Son)** | **0.9473** | **0.9379** | **0.9333** |

---

## 6. Latency Benchmarkları

### 6.1 CPU E2E Gecikme (Smoke Video)

> **Kaynak:** `reports/detector_v5_e2e_latency_benchmark.json`

### Elde Edilen Performans Kazançları (Ortalama E2E Süreleri)
| Mod | Optimizasyon Öncesi | Optimizasyon Sonrası | Yarışma Hedefi | Durum |
|---|:---:|:---:|:---:|---|
| **GPU (CUDA Fallback)** | 9.68 sn | **1.52 sn** | 8.0 sn | **GEÇTİ** ✅ |
| **CPU (CPU EP)** | 9.84 sn | **2.24 sn** | 8.0 sn | **GEÇTİ** ✅ |

> **Önemli Not:** Bu benchmark **detector_v5 ile güncellenmiş** sistemin optimize edilmiş CPU/GPU performansını yansıtır. Yapılan optimizasyonlar (lazy loading, CPU warmup bypass ve thread optimizasyonları) sayesinde E2E latency 8 saniyelik limitin altına çekilmiştir.

### 6.2 Tekli Frame ONNX CPU Gecikme

| Model | CPU Gecikme |
|-------|:---:|
| detector_v5 ONNX (FP16) | 118.68 ms/frame |

---

## 7. OCR Doğruluk Metrikleri

> **Kaynak:** `reports/ocr_accuracy_test_results.json`

| Metrik | Değer | Test Seti |
|--------|-------|-----------|
| Exact Match Rate | 1.00 (100%) | 14 sentetik plaka |
| Mean CER | 0.00 | 14 sentetik plaka |
| Latency (CRNN) | ~250.8 ms | `plate_ocr.py:LATENCY_MS` |

> ⚠️ **Uyarı:** Test seti yalnızca `34TR1001`–`34TR1014` formatında sentetik görüntülerden oluşmaktadır. Gerçek dünya plaka varyasyonları (kirli, gece, eğimli, bulanık) ile doğrulama yapılmamıştır.

---

## 8. Pytest Birim Test Sonuçları

```
platform win32 — Python 3.14.3, pytest-9.0.3
collected 84 items
84 passed in 100.01s
```

| Test Modülü | Test Sayısı | Durum |
|-------------|:-----------:|-------|
| test_all_components.py | 5 | ✅ |
| test_auth_utils.py | 2 | ✅ |
| test_bff_media_boundary.py | 2 | ✅ |
| test_competition_contract.py | 6 | ✅ |
| test_dataset_corpus.py | 3 | ✅ |
| test_latest_frame.py | 1 | ✅ |
| test_model_manifest.py | 2 | ✅ |
| test_ocr_accuracy.py | 8 | ✅ |
| test_per_class_regression.py | 10 | ✅ |
| test_qod_orchestrator.py | 4 | ✅ |
| test_signaling_api.py | 3 | ✅ |
| test_stress.py | 5 | ✅ |
| test_telemetry_hub.py | 2 | ✅ |
| test_tracking_pipeline.py | 2 | ✅ |
| test_webrtc_signaling.py | 5 | ✅ |

---

## 9. Model Kilit (SHA-256) Doğrulaması

> **Kaynak:** `model_lock.json`

| Model | SHA-256 | Durum |
|-------|---------|-------|
| detector.onnx | `da1840...a966d5b9` | ✅ Kilitli |
| detector_optimized.onnx | `da1840...a966d5b9` (özdeş) | ✅ Kilitli |
| yolov8n.onnx (COCO) | `edb369...08fede4` | ✅ Kilitli |
| crnn.onnx | `f95f0b...9ce3cc53` | ✅ Kilitli |
| lprnet.onnx | `6c8ff0...383aa50` | ✅ Kilitli |
| cnn_lstm.onnx | `c432ce...09f3039` | ✅ Kilitli (HEDEF model) |

---

## 10. Eksik / Kanıtsız Metrikler

| Metrik | Durum | Neden Eksik |
|--------|-------|-------------|
| T4 GPU gecikme | ❌ HEDEF | Gerçek T4 ortamı erişimi yok |
| Gerçek plaka OCR CER | ❌ HEDEF | Gerçek plaka veri seti etiketlenmemiş |
| Canlı 5G QoD testi | ❌ HEDEF | Turkcell CAMARA SIM gerekli |
| Android WebRTC RTP testi | ❌ HEDEF | İki uç ortam gerekli |
| mAP50-95 için sınıf bazlı güven aralığı | ❌ Hesaplanmamış | Bootstrap/cross-val gerekli |

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
