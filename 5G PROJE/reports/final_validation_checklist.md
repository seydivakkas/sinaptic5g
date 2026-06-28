# Final Doğrulama Çizelgesi — SİNAPTİC5G FTR Sistemi

**Proje:** SİNAPTİC5G — 5G Destekli Akıllı Yol Güvenliği (FTR Yarışması)  
**Tarih:** 2026-06-26  
**Faz:** 5 — Test & Doğrulama  

---

## ✅ Üretim Kilidi Doğrulaması

| Kontrol | Komut | Beklenen |
|---------|-------|---------|
| model_lock.json SHA256 | `python ftr_main.py --verify-lock` | OK |
| detector.onnx değişmedi | `python -m pytest tests/test_per_class_regression.py::test_no_model_file_mutation` | PASS |
| crnn.onnx değişmedi | `python -m pytest tests/test_ocr_accuracy.py::test_production_crnn_not_modified` | PASS |
| FP16 export kilidi | `python scripts/export_fp16_onnx.py --self-test` | OK |

---

## ✅ Regresyon Testleri

```bash
python -m pytest tests/test_per_class_regression.py -v
```

| Test | Beklenen |
|------|---------|
| test_production_model_exists | PASS |
| test_model_produces_output | PASS |
| test_per_class_recall[telefonla_konusma] | recall ≥ 0.40 |
| test_per_class_recall[teknocan] | recall ≥ 0.25 |
| test_per_class_recall[bilgisayar] | recall ≥ 0.25 |
| test_per_class_recall[emniyet_kemeri_ihlali] | recall ≥ 0.30 |
| test_per_class_recall[license_plate] | recall ≥ 0.65 |

---

## ✅ OCR Doğruluk Testleri

```bash
python -m pytest tests/test_ocr_accuracy.py -v
```

| Test | Beklenen |
|------|---------|
| test_crnn_model_loads | PASS |
| test_exact_match_rate | ≥ 60% |
| test_mean_cer | ≤ 15% |
| test_no_null_predictions | boş tahmin < 50% |
| test_plate_format_validity | geçerli format ≥ 50% |

---

## ✅ Stres Testleri

```bash
python -m pytest tests/test_stress.py -v
```

| Test | Beklenen |
|------|---------|
| test_tracker_buffer_no_memory_leak | aktif track ≤ 100 |
| test_ocr_buffer_cleanup | stale buffer temizlendi |
| test_debounce_counter_cleanup | stale track counter silindi |
| test_adaptive_stride_range | stride beklenen aralıkta |
| test_500_frame_synthetic_pipeline | hata yok, bellek < 200 MB |
| test_production_model_lock_integrity | SHA256 eşleşiyor |

---

## ✅ Augmentasyon Doğrulaması

```bash
# Etiket kalite denetimi (smoke-run):
python scripts/audit_labels.py \
    --images-dir dataset/images/train \
    --labels-dir dataset/labels/train \
    --max-images 50 \
    --output-csv reports/suspicious_labels.csv

# Augmentasyon pipeline smoke-run:
python scripts/augment_dataset.py \
    --images-dir dataset/images/train \
    --labels-dir dataset/labels/train \
    --output-dir dataset_augmented/images/train \
    --n-augment 1
```

| Kontrol | Beklenen |
|---------|---------|
| Şüpheli etiket oranı | < 30% |
| Augmentasyon çıktısı | boyutu arttı |
| Orijinal dataset değişmedi | ✅ |

---

## ✅ Pipeline Optimizasyon Doğrulaması

```bash
# Adaptif stride testi:
python -c "from ftr_main import compute_adaptive_stride; print(compute_adaptive_stride(True, False))"

# Plaka voting testi:
python -c "
from plate_ocr import PlateRecognizer
p = PlateRecognizer()
print('buffer_size:', p.voting_buffer_size, 'edit_dist:', p.edit_distance_threshold)
"
```

| Kontrol | Beklenen |
|---------|---------|
| compute_adaptive_stride import | OK |
| PlateRecognizer voting_buffer=7 | OK |

---

## ✅ Docker Boyut Kontrolü

```bash
# Proje dosya boyutu (Docker olmadan):
python scripts/check_docker_size.py --check-files --check-models
```

| Kontrol | Beklenen |
|---------|---------|
| Proje dosya boyutu | < 8 GB |
| Model boyutları | ±20% referans değeri |

---

## ✅ Benchmark Doğrulaması

```bash
# CPU dry-run benchmark:
python scripts/benchmark_t4_cuda.py --dry-run --all-models
```

| Model | CPU Hedef | T4 GPU Hedef |
|-------|-----------|-------------|
| detector.onnx | < 500 ms | < 100 ms |
| crnn.onnx | < 200 ms | < 50 ms |

---

## ✅ FTR Pipeline End-to-End Doğrulaması

```bash
python ftr_main.py --video test_data/sample_video.mp4 --output /tmp/test_results.json
```

| Kontrol | Beklenen |
|---------|---------|
| JSON çıktısı oluştu | ✅ |
| Çalışma süresi < 8s/video-dakikası | ✅ |
| Exception yok | ✅ |
| model_lock.json SHA256 tutarlı | ✅ |

---

## ✅ Üretilen Dosyalar

### Faz 2 — Veri Seti Güçlendirme
- [x] `scripts/augment_dataset.py`
- [x] `scripts/copy_paste_augment.py`
- [x] `scripts/audit_labels.py`

### Faz 3 — Model Eğitimi Hazırlığı
- [x] `scripts/train_crnn_tr_plate.py`
- [x] `scripts/train_temporal_lstm.py`
- [x] `reports/model_training_plan.md`

### Faz 4 — Pipeline Optimizasyonu
- [x] `src/tracking_pipeline.py`
- [x] `ftr_main.py`
- [x] `plate_ocr.py`
- [x] `scripts/export_fp16_onnx.py`

### Faz 5 — Test & Doğrulama
- [x] `tests/test_per_class_regression.py`
- [x] `tests/test_ocr_accuracy.py`
- [x] `tests/test_stress.py`
- [x] `scripts/check_docker_size.py`
- [x] `scripts/benchmark_t4_cuda.py`
- [x] `reports/final_validation_checklist.md`

---

## Üretim Güvenliği Özeti

| Kural | Durum |
|-------|-------|
| detector_v5 kilitli | ✅ |
| Eski referans (v2/v3/v4) yok | ✅ |
| Her script salt okunur model kullanımı | ✅ |
| `model_lock.json` korumalı | ✅ |

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
amaçlı olarak paylaşılmıştır.

YASAKLAR:
  1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
  2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
  3. Alt lisanslanamaz, satılamaz veya devredilemez.
  4. Tersine mühendislik yapılamaz.

İZİN VERİLEN KULLANIM:
  - GitHub üzerinde görüntüleme ve okuma.
  - Kişisel öğrenim amacıyla kodu inceleme (kopyalanadan).

YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
İzin talepleri için: GitHub @seydivakkas
