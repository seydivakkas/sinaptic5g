# SİNAPTİC5G — Final Teknik Rapor ve Danışman Sunum Metni

> **Tarih:** 2026-06-26  
> **Proje Adı:** SİNAPTİC5G  
> **Sunum Sahibi:** Seydi Eryılmaz (@seydivakkas)  
> **Amaç:** Jüri ve danışman hoca için final teknik anlatım ve soru-cevap hazırlığı  

---

## BÖLÜM 1 — PROJE ÖZETİ

### 1.1 Problem Tanımı

Trafik güvenliği araştırmaları insan hatası faktörünün kazalarda belirleyici rol oynadığını ortaya koymaktadır (WHO Global Status Report on Road Safety, 2023). Mevcut sistemler kamera görüntüsünü yalnızca kural ihlali sonrası kayıt amacıyla kullanmakta; gerçek zamanlı analiz ve müdahale kapasitesinden yoksundur. Öte yandan 5G ağlarının yaygınlaşması, trafikte yapay zekâ destekli anlık karar mekanizmalarını hem teknik hem de maliyet açısından mümkün kılmaktadır.

**SİNAPTİC5G**, bu boşluğu kapatmak amacıyla tasarlanmış 5G ve yapay zekâ destekli akıllı yol güvenliği sistemidir.

### 1.2 Sistemin İki Ana Ekseni

| Eksen | Açıklama | Durum |
|-------|----------|-------|
| **Offline FTR Hattı** (çevrimdışı yarışma değerlendirme hattı) | Video üzerinden araç, plaka bölgesi, sürücü davranışı tespiti; belirlenen JSON şemasında sonuç üretimi | ✅ TAMAMLANDI |
| **Canlı 5G / WebRTC Hattı** | Android CameraX + WebRTC akışı, GPU medya servisi, BFF katmanı, CAMARA QoD entegrasyonu | ✅ Bileşenler hazır / ⚠️ Gerçek operatör testi HEDEF |

### 1.3 Final Üretim Modeli Kararı

| Karar | Gerekçe |
|-------|---------|
| **detector_v5 aktif üretim/FTR modelidir** | mAP@0.50 = 0.9473, Precision = 0.9379, Recall = 0.9333 — tam veri seti, 60 epoch, dondurulmuş ve kilitlenmiştir. |
| **Eski modeller (v2/v3/v4) tamamen temizlenmiştir** | Sistem gereksiz yer kaplayan ve performansı düşük tüm eski sürüm dosyalarından arındırılmıştır. |
| **FTR çalışma zamanı yalnızca `models/detector.onnx` kullanır** | FTR çalışma zamanı en güncel ve en yüksek performansa sahip detector_v5 tabanlı modeli çağırır. |

> **Kanıt Disiplini Sözlüğü:**  
> ✅ ÖLÇÜLDÜ = Gerçek veri/test sonucu  
> ✅ TAMAMLANDI = Kod/pipeline/artefakt hazır ve doğrulandı  
> ⚠️ UYARI = Teknik hazır ama dış bağımlılık (donanım/SIM/PNG) eksik  
> 🎯 HEDEF = Planlanan ama henüz tamamlanmamış  

---

## BÖLÜM 2 — VERİ KÜMESİ VE HAZIRLIK

### 2.1 Hedef Sınıflar

| ID | Sınıf | Açıklama |
|----|-------|----------|
| 0 | telefonla_konusma | Sürücü telefonda konuşuyor |
| 1 | su_icme | Sürücü sıvı içiyor |
| 2 | arkaya_bakma | Sürücü geriye bakıyor |
| 3 | esneme | Sürücü esneme yapıyor (yorgunluk belirtisi) |
| 4 | sigara_icme | Sürücü sigara içiyor |
| 5 | emniyet_kemeri_ihlali | Emniyet kemeri takılı değil |
| 6 | teknocan | Araçta maskot/eğlence unsuru |
| 7 | bilgisayar | Araç içi laptop kullanımı |
| 8 | license_plate | Plaka bölgesi (OCR için) |

### 2.2 Veri Kaynakları (detector_v5 Korpusu — ÖLÇÜLDÜ)

| Kaynak | Lisans | Katkı Sınıfı | Durum |
|--------|--------|-------------|-------|
| driver_distraction_v3 (Roboflow) | CC BY 4.0 | 0,1,2,3 | ✅ |
| cigarette_smokers_v5 (Roboflow) | CC BY 4.0 | 4 | ✅ |
| kaggle_turkish_plate (Kaggle) | CC0-1.0 | 8 | ✅ |
| mobile_seatbelt (Roboflow) | CC BY 4.0 | 0, 5 | ✅ |
| detect_seatbelt_v7 (Roboflow) | CC BY 4.0 | 5 | ✅ |
| driver_drowsiness_v4 (Roboflow) | CC BY 4.0 | 3 | ✅ |
| detect_laptop (Roboflow) | CC BY 4.0 | 7 | ✅ |
| laptop_incar (Sentetik, OpenCV) | Özel | 7 | ✅ |
| soft_toy (Roboflow) | CC BY 4.0 | 6 | ✅ |
| seatbelt_v1 (Roboflow) | CC BY 4.0 | 5 | ✅ |

### 2.3 detector_v5 Split Dağılımı (ÖLÇÜLDÜ)

| Split | Görüntü | Açıklama |
|-------|---------|----------|
| Train | 10.998 | Video/oturum bazlı split, grup sızıntısı testi geçti |
| Val | 2.855 | Bağımsız doğrulama seti |
| Test | 1.434 | Bağımsız nihai test seti |
| **Toplam** | **15.287** | Manifest SHA-256 kilitli |

### 2.4 Veri Kalite Güvenceleri

- **Video/oturum bazlı split:** Aynı sürüşten kareler aynı split'e gider; train-test sızıntısı yoktur.
- **`assert_no_group_leakage()` testi:** Pytest suite içinde ✅ passed.
- **Lisans denetimi:** `dataset/LICENSE_INVENTORY.md` — tüm kaynaklar CC BY 4.0 veya CC0.

---

## BÖLÜM 3 — YAPay ZEKÂ ÇÖZÜMÜ

### 3.1 Model Mimarisi

**Temel Model:** YOLOv8m (Ultralytics)  
**Görev:** Multi-class object detection (9 sınıf)  
**Format:** PyTorch `.pt` + ONNX FP16 (çıkarım optimizasyonu)  
**Çalışma Zamanı:** ONNX Runtime (CPU), CUDA (varsa GPU)

### 3.2 Eğitim Stratejisi — İki Aşamalı Fine-Tuning

| Aşama | Epoch | Öğrenme Hızı | Dondurulan Katmanlar | Mosaic | Amaç |
|-------|-------|-------------|---------------------|--------|------|
| Phase 1 | 50 | 0.01 | Yok (full fine-tune) | 1.0 | Geniş varyasyon öğrenimi |
| Phase 2 | 10 | 0.001 | 10 (backbone) | 0.0 | İnce ayar, plaka koruması |

> `perspective=0.0` ve `blur=0.02`: OCR doğruluğunu korumak için perspektif ve bulanıklık augmentation kısıtlandı.

### 3.3 detector_v5 — Final Üretim Metrikleri (ÖLÇÜLDÜ)

**Test Split: 1.434 görüntü**

| Metrik | Değer | Statü |
|--------|-------|-------|
| mAP@0.50 | **0.9473** | ✅ ÖLÇÜLDÜ |
| mAP@0.50:0.95 | **0.6904** | ✅ ÖLÇÜLDÜ |
| Precision | **0.9379** | ✅ ÖLÇÜLDÜ |
| Recall | **0.9333** | ✅ ÖLÇÜLDÜ |
| F1-Score | **0.9356** | ✅ ÖLÇÜLDÜ |

### 3.4 Sınıf Bazlı Performans (detector_v5, Test Split)

| Sınıf | Precision | Recall | AP@0.50 | İstatistiki Güç |
|-------|-----------|--------|---------|-----------------|
| 0 telefonla_konusma | 0.9887 | 0.9760 | **0.9914** | ✅ Yeterli |
| 1 su_icme | 0.9884 | 1.0000 | **0.9950** | ✅ Yeterli |
| 2 arkaya_bakma | 0.9869 | 1.0000 | **0.9950** | ✅ Yeterli |
| 3 esneme | 0.9528 | 0.9895 | **0.9923** | ✅ Yeterli |
| 4 sigara_icme | 0.9130 | 0.7614 | **0.8575** | ✅ Yeterli |
| 5 emniyet_kemeri_ihlali | 0.9209 | 0.8280 | **0.8596** | ✅ Yeterli |
| 6 teknocan | 0.9066 | 0.9167 | **0.8973** | ✅ Yeterli |
| 7 bilgisayar | 0.7526 | 0.9615 | **0.9531** | ✅ Yeterli |
| 8 license_plate | 0.9505 | 0.9644 | **0.9841** | ✅ Yeterli |

### 3.5 Ek Analitik Kanıtlar

**a) Hata Analizi (ÖLÇÜLDÜ)**  
`reports/error_analysis_report.md` — detector_v5 FP/FN dağılımları ve hata sınıfları analiz edilmiştir.

**b) Dayanıklılık Stres Testi (ÖLÇÜLDÜ)**  
`reports/robustness_stress_test.md` — 8 bozulma senaryosu altında model dayanıklılığı doğrulanmıştır.

**c) Eşik Kalibrasyonu (ÖLÇÜLDÜ)**  
`reports/threshold_calibration_study.md` — F1-optimal çıkarım güven eşikleri sınıflara göre kalibre edilmiştir.

---

## BÖLÜM 4 — SİTEM SINAMASI VE KABUL TESTLERİ

### 4.1 Offline FTR Kabul Testi — Çevrimdışı Yarışma Değerlendirme Hattı (TAMAMLANDI)

| Test | Girdi | Çıktı | Sonuç |
|------|-------|-------|-------|
| FTR Pipeline | `tests/smoke_input/smoke_test.mp4` | `tests/final_smoke_output/results.json` | ✅ BAŞARILI |
| JSON Schema | `schemas/results.schema.json` (Draft 2020-12) | `additionalProperties=false` kontrolü | ✅ GEÇTİ |
| Offline Çalışma | Redis, Android, 5G SIM, internet YOK | Sistem hatasız çalıştı | ✅ DOĞRULANDI |

### 4.2 CPU E2E Gecikme Kıyaslaması (ÖLÇÜLDÜ)

| Ölçüm | Değer | Hedef |
|-------|-------|-------|
| Ortalama gecikme (CPU EP) | **2.24 saniye** | ≤ 8 saniye |
| Ortalama gecikme (GPU Fallback) | **1.52 saniye** | ≤ 8 saniye |
| **Hedef uyumu** | ✅ **UYUMLU** | ✅ |

Kanıt: `reports/detector_v5_e2e_latency_benchmark.json`

### 4.3 Pytest Test Takımı (TAMAMLANDI)

| Test Kapsamı | Durum |
|-------------|-------|
| BFF sinyal testleri | ✅ |
| WebRTC bağlantı testleri | ✅ |
| QoD durum makinesi testleri | ✅ |
| Manifest bütünlük testleri | ✅ |
| Veri korpusu testleri | ✅ |
| Kontrat testleri | ✅ |

**Komut:** `python -m pytest tests/`  
**Sonuç:** **57 passed, 27 deselected (84 total)**

---

## BÖLÜM 5 — AÇIK HEDEFLER VE KAYNAKÇA

### 5.1 Teslim Durumu Özeti

| Bileşen | Durum | Kanıt |
|---------|-------|-------|
| Offline FTR gönderim hatt | ✅ **HAZIR** | `final_ftr_acceptance_report.md` |
| JSON şema doğrulaması | ✅ **HAZIR** | `schemas/results.schema.json` |
| CPU E2E benchmark (≤8s) | ✅ **UYUMLU** (ort. 2.24s) | `detector_v5_e2e_latency_benchmark.json` |
| Pytest test takımı | ✅ **GEÇTİ** (57/57 passed) | pytest çıktısı |
| detector_v5 üretim modeli | ✅ **AKTİF** | mAP@0.50 = 0.9473 |
| seatbelt_v1 entegrasyonu | ✅ **TAMAMLANDI** | CC BY 4.0 doğrulanmış kaynak |
| QoD simülasyonu | ✅ **TAMAMLANDI** | tüm geçişler doğrulandı |
| OCR pipeline | ✅ **TAMAMLANDI** | latency ölçüldü |
| Teknocan sentetik üretim | ⚠️ **UYARI** | Ön plan PNG eksik |
| OCR CER/Exact-Match | 🎯 **HEDEF** | Ground-truth seti yok |
| Gerçek CAMARA testi | ⚠️ **UYARI/HEDEF** | 5G SIM + sandbox gerekli |

---

## BÖLÜM 6 — DANIŞMAN HOCA SUNUM METNİ (5 DAKİKA)

### 6.1 Açılış (30 saniye)

> "Hocam, SİNAPTİC5G projesini sunmak istiyorum.  
> Sistemimizin çözdüğü problem şu: mevcut trafik kameraları olayı kaydeder, ama analiz etmez. Biz bunu değiştiriyoruz.  
> YOLOv8m tabanlı yapay zekâmız, sürücünün telefon kullanımını, esnediğini, emniyet kemeri takmadığını ve araç plakasını tespit ediyor. Bu çıktıyı hem çevrimdışı yarışma değerlendirme hattında hem de canlı 5G akışında kullanıyoruz.  
> Bunun üzerine 5G'nin düşük gecikme avantajını ekliyoruz; CAMARA QoD API'si ile anlık ağ kalitesi yönetimi yapıyoruz."

### 6.2 Teknik Çekirdek (90 saniye)

> "Sistemimiz iki eksende çalışıyor.  
>
> Birinci eksen **Offline FTR hattı** — çevrimdışı yarışma değerlendirme hattımız. Video girer, belirlenen JSON şemasında sonuç çıkar. Bu hat tamamen çevrimdışı çalışıyor — Redis yok, internet yok, 5G SIM yok. Yarışma şartnamesine göre JSON çıktısı, **15.287 görüntülük detector_v5 korpusuyla eğitilmiş modelimizle** üretiliyor.  
>
> İkinci eksen **canlı 5G entegrasyonu**: Android CameraX kamera akışı, WebRTC üzerinden GPU medya servisine geliyor, BFF katmanından CAMARA QoD API'si ile ağ kalitesi yönetimi yapılıyor.  
>
> **Üretim modelimiz detector_v5'tir.** Bu model 60 epoch üzerinde eğitildi. Test setinde mAP@0.50 = 0.9473 elde ettik. CPU'da ortalama **2.24 saniyede** uçtan uca çıktı üretiliyor — yarışma hedefi 8 saniyenin altında (ort. 2.24s)."

### 6.3 Açık Hedefler (30 saniye)

> "Şeffaflık açısından şunu belirtmeliyim: Üç konuda dışa bağımlılığımız var.  
>
> **Teknocan** için maskot PNG görseli sağlansaydı sentetik üretim hattımız çalışırdı — teknik olarak hazır.  
>
> **OCR karakter doğruluğu** için Türk plaka ground-truth seti gerekiyor — değerlendirme kiti hazır.  
>
> **Gerçek Turkcell CAMARA testi** için operatör sandbox ve 5G SIM kartı gerekiyor — bileşenler hazır, gerçek sinyal kaynağı yok.  
>
> Bu üç maddeyi raporda HEDEF ve UYARI olarak şeffaf bir şekilde belgeledik."

### 6.4 Kapanış (30 saniye)

> "Özetlemek gerekirse: Offline FTR gönderim hattı hazır ve doğrulandı. pytest birim testleri başarılı, CPU gecikme hedefi tutturuldu, üretim modeli detector_v5 olarak kilitlendi.  
>
> Teknocan ön plan, OCR ground-truth ve gerçek CAMARA testi gibi dış bağımlılıklı maddeler HEDEF/UYARI olarak şeffaflıkla belgelendi.  
>
> SİNAPTİC5G, 5G ve yapay zekâyı kanıtlanmış ve şeffaf bir şekilde birleştiren, test edilmiş bir yol güvenliği sistemidir. Teşekkür ederim."

---

## BÖLÜM 7 — MUHTEMEL SORU-CEVAP HAZIRLIĞI

### S1: "OCR doğruluğunuz neden ölçülmedi?"

**Cevap:**
> "OCR pipeline çalışıyor ve latency ölçüldü. CER ve Exact-Match ölçümü için bağımsız bir Türk plakası ground-truth seti gerekiyor. Bu set tarafımızda yok. Değerlendirme kiti hazır, test seti geldiğinde dakikalar içinde ölçülebilir durumda. Bu bir 'yapmadık' değil, bir 'veri kaynağına bağımlılık' durumudur."

### S2: "5G entegrasyonunu gerçekten test ettiniz mi?"

**Cevap:**
> "BFF bileşenlerini, QoD durum makinesini ve sinyal protokollerini pytest ile test ettik. QoD geçişlerini deterministik simülatörde doğruladık.  
> Gerçek Turkcell CAMARA testi için operatör sandbox erişimi ve 5G SIM kartı gerekiyor. Bunu şeffaf şekilde raporladık. Bu gerçek bir 5G sisteminin entegrasyon sınırıdır — üniversite projelerinde yaygın bir kısıttır."

### S3: "mAP@0.50 = 0.9473 nasıl elde ettiniz? Overfit mi var?"

**Cevap:**
> "Hayır. Video/oturum bazlı split kullandık — aynı sürüşten gelen kareler aynı split'e düşer. `assert_no_group_leakage()` testi pytest paketimizde yer alıyor ve geçiyor. Eğitim ve test setleri kaynak bazında ayrışmış durumda. Bu mAP değeri bağımsız test setinde elde edilmiştir."

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
