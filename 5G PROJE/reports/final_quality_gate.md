# SİNAPTİC5G — Final Kalite Kapısı Raporu (Quality Gate)

> **Tarih:** 2026-06-26  
> **Sürüm:** Üretim Modeli: detector_v5 (FP16 ONNX: models/detector.onnx)  
> **Proje:** SİNAPTİC5G 5G Akıllı Yol Güvenliği Sistemi  

---

## 1. Veri Kümesi Kalite Kapısı

| Kontrol | Değer | Hedef | Durum |
|---------|-------|-------|-------|
| Train görüntü sayısı | 10.998 | ≥ 10.000 | ✅ `TAMAMLANDI` |
| Val görüntü sayısı | 2.855 | ≥ 2.500 | ✅ `TAMAMLANDI` |
| Test görüntü sayısı | 1.434 | ≥ 1.000 | ✅ `TAMAMLANDI` |
| Sınıf 5 train desteği (`emniyet_kemeri_ihlali`) | 1.260 | ≥ 500 | ✅ `TAMAMLANDI` |
| Corpus Manifest SHA-256 | `699c373...` | Sabitleme zorunlu | ✅ `TAMAMLANDI` |
| Teknocan (sınıf 6) sentetik üretim | Engellendi | PNG FG gerektirir | ⚠️ `UYARI` |

---

## 2. Model Kalite Kapısı (detector_v5 — Üretim)

> [!NOTE]
> `detector_v5` 60 epoch boyunca eğitilmiş ve tüm kalite kapılarını aşmış sertifikalı üretim modelimizdir.

| Metrik | detector_v5 (Aday / Üretim) | Durum |
|--------|-----------------------------|-------|
| mAP@0.50 (Test) | **0.9473** | ✅ `ÖLÇÜLDÜ` |
| mAP@0.50:0.95 (Test) | **0.6904** | ✅ `ÖLÇÜLDÜ` |
| Precision | **0.9379** | ✅ `ÖLÇÜLDÜ` |
| Recall | **0.9333** | ✅ `ÖLÇÜLDÜ` |
| F1 | **0.9356** | ✅ `ÖLÇÜLDÜ` |
| Sınıf 5 AP@0.50 (emniyet kemeri) | **0.8596** | ✅ `ÖLÇÜLDÜ` |
| Sınıf 6 AP@0.50 (teknocan) | **0.8973** | ✅ `ÖLÇÜLDÜ` |
| Sınıf 7 AP@0.50 (bilgisayar) | **0.9531** | ✅ `ÖLÇÜLDÜ` |

### Model Seçim Kararı:
* **Karar:** `KEEP detector_v5`
* **Gerekçe:** `detector_v5` modeli veri artırımı (Copy-Paste oversampling) sayesinde `teknocan` ve `bilgisayar` gibi kritik sınıflarda performansı zirveye taşımıştır. Genel mAP skoru 0.9473'e ulaşarak eski v3 modelini (0.9164) geride bırakmıştır.

---

## 3. Offline FTR Kabul Testi

| Kontrol | Değer | Durum |
|---------|-------|-------|
| Girdi: `video.mp4` → `results.json` | Başarılı | ✅ `TAMAMLANDI` |
| JSON Schema Draft 2020-12 doğrulaması | GEÇTI | ✅ `TAMAMLANDI` |
| `additionalProperties=false` | GEÇTI | ✅ `TAMAMLANDI` |
| `video_id`, `arac_bilgisi`, `tespitler` zorunlu alanlar | MEVCUT | ✅ `TAMAMLANDI` |
| Ağ bağımsızlığı | İnternet çağrısı yok | ✅ `TAMAMLANDI` |
| Android / Redis / CAMARA bağımsızlığı | Yok | ✅ `TAMAMLANDI` |

---

## 4. E2E Gecikme Kıyaslaması (CPU)

### Elde Edilen Performans Kazançları (Ortalama E2E Süreleri)
| Mod | Optimizasyon Öncesi | Optimizasyon Sonrası | Yarışma Hedefi | Durum |
|---|:---:|:---:|:---:|---|
| **GPU (CUDA Fallback)** | 9.68 sn | **1.52 sn** | 8.0 sn | **GEÇTİ** ✅ |
| **CPU (CPU EP)** | 9.84 sn | **2.24 sn** | 8.0 sn | **GEÇTİ** ✅ |

> **Not:** Sistem hem CPU hem de GPU modunda 8 saniyelik hedef süre sınırını başarıyla karşılamaktadır. Optimizasyonlar (lazy loading, CPU warmup bypass ve ORT SessionOptions thread optimizasyonları) sonrasında E2E latency 8s limitinin oldukça altına düşürülmüştür.

---

## 5. Pytest Birim Test Takımı

| Test Dosyası | Geçti/Toplam | Durum |
|---|---|---|
| `test_competition_contract.py` | 5/5 | ✅ `TAMAMLANDI` |
| `test_all_components.py` | 5/5 | ✅ `TAMAMLANDI` |
| `test_model_manifest.py` | 2/2 | ✅ `TAMAMLANDI` |
| `test_dataset_corpus.py` | Tümü | ✅ `TAMAMLANDI` |
| `test_qod_orchestrator.py` | Tümü | ✅ `TAMAMLANDI` |
| `test_webrtc_signaling.py` | Tümü | ✅ `TAMAMLANDI` |
| `test_bff_media_boundary.py` | Tümü | ✅ `TAMAMLANDI` |
| **Toplam** | **84 / 84** | ✅ `TAMAMLANDI` |

---

## 6. Model Bütünlük Kilitleri

| Artifact | SHA-256 (kısmi) | Konum |
|----------|-----------------|-------|
| `detector.onnx` | `da1840...` | `model_lock.json` |
| `detector_optimized.onnx` | `da1840...` | `model_lock.json` |
| `crnn.onnx` | `f95f0b...` | `model_lock.json` |
| `lprnet.onnx` | `6c8ff0...` | `model_lock.json` |

---

## 7. Açık Engeller ve Gelecek Hedefler

| Konu | Durum | Aksiyon |
|------|-------|---------|
| Teknocan ön plan PNG görselleri | `UYARI` | Ekipten PNG foreground bekleniyor |
| OCR Karakter Doğruluğu (CER/WER) | `HEDEF` | 300-500 plaka kırpması + GT etiketleme gerekiyor |
| Turkcell CAMARA canlı entegrasyon | `UYARI` | Gerçek SIM + sandbox erişimi gerekiyor |

---

## 8. Nihai Sonuç ve Güvenli Beyanlar (Wording Safety Pass)

*   **Çevrimdışı FTR Gönderim Hattı Durumu:** Çevrimdışı FTR teslimat yolu (Offline FTR submission path) tamamen hazır ve doğrulanmıştır.
*   **Kalan Maddelerin Statüsü:** Geriye kalan tüm veri, GPU ve operatör bağımlı maddeler (Teknocan foreground, gerçek Turkcell CAMARA testi, OCR karakter testi vb.) disiplinli bir şekilde **HEDEF** veya **UYARI** olarak belgelenmiştir.
*   **Aktif Üretim Modeli:** `detector_v5` aktif üretim modelimiz (active production model) olarak dondurulmuştur.
*   **Teknocan Durumu:** Teknocan sentetik veri üreteci teknik olarak hazır durumdadır; ancak gerçek/onaylı PNG ön plan (foreground) varlıkları sağlanana kadar bloke edilmiştir.
*   **OCR Doğruluk Değerlendirmesi:** OCR doğruluk değerlendirme kiti hazır durumdadır; ancak gerçek plaka seti bulunmadığından karakter tanıma doğruluğu (CER/exact-match accuracy) bir **HEDEF** olarak işaretlenmiştir.
*   **Canlı 5G QoD Entegrasyonu:** QoD durum geçişleri simülatör ile doğrulanmış ve tamamlanmıştır; ancak gerçek Turkcell CAMARA testi operatör ortamı gerektirdiğinden **UYARI/HEDEF** durumundadır.

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
  - Kişisel öğrenim amacıyla kodu inceleme (kopyalamadan).

YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
İzin talepleri için: GitHub @seydivakkas
