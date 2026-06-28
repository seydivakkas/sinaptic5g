# SİNAPTİC5G — Sistem Mimarisi Özet Raporu

> **Sürüm:** 1.0  
> **Tarih:** 2026-06-25  
> **Yazar:** Seydi Eryılmaz (@seydivakkas)

---

## 1. Sistem Amacı ve Problem Tanımı

SİNAPTİC5G, TEKNOFEST yarışması kapsamında geliştirilen, **5G ağ teknolojisi ile yapay zekâyı entegre eden akıllı yol güvenliği analiz sistemidir**. Sistemin temel problemi şu şekilde özetlenebilir: Bir araç içi kameradan alınan video akışında, sürücünün tehlikeli davranışlarını, dikkat dağıtıcı nesneleri ve araç kimliğini gerçek zamanlı veya çevrimdışı olarak tespit edip raporlamak.

### Problem Bağlamı

Geleneksel araç güvenliği sistemleri, tespit ve karar mantığını araca gömülü donanımda çalıştırır. Bu yaklaşım, güncelleme esnekliğinden yoksundur ve hesaplama kapasitesi sınırlıdır. SİNAPTİC5G, aşağıdaki özgün yaklaşımı benimsemektedir:

- **5G + MEC (Multi-Access Edge Computing) entegrasyonu:** Ağır hesaplamaların (YOLOv8m tespiti) kenar sunucusunda (GPU Media Service) yapılması, Android cihaza yalnızca hafif çıkarım görevinin verilmesi.
- **CAMARA QoD (Quality on Demand) API:** Ağ koşulları kötüleştiğinde veya kritik tespit gerçekleştiğinde 5G ağ kaynaklarının öncelikli tahsisi için Turkcell CAMARA API entegrasyonu.
- **Çevrimdışı FTR (Final Test Run) Modu:** İnternet bağlantısı olmaksızın Docker container içinde çalışan, şema-doğrulamalı ve tekrarlanabilir yarışma gönderim hattı.

### Ana Çıktı Hedefi

Sistem şu uçtan uca akışı gerçekleştirir:

```
Video Girdisi → Kare Örnekleme → Ön İşleme → YOLO Tespiti → BoTSORT Takip
→ OCR/Plaka Okuma → Temporal Karar → Risk Skoru → JSON Çıktısı
```

Nihai ürün: Yarışma şemasına uygun `results.json` dosyası.

---

## 2. Genel Sistem İşleyişi — Uçtan Uca Pipeline

| Adım | Sorumlu Bileşen | Dosya | Girdi | Çıktı |
|------|----------------|-------|-------|-------|
| 1. Video Girişi | OpenCV VideoCapture | `ftr_main.py` | MP4 dosyası | Frame dizisi |
| 2. Adaptif Stride | `compute_adaptive_stride()` | `ftr_main.py:81` | FPS, tespit durumu, GPU flag | Sonraki örnekleme zamanı (sn) |
| 3. Parlaklık Kontrolü & Zero-DCE | `zero_dce_enhance()` | `src/models/low_light/zero_dce.py` | Ham kare | Işık-iyileştirilmiş kare |
| 4. Lens Düzeltme | `cv2.undistort()` | `ftr_main.py:504` | Kare + K,D matrisleri | Distorsiyonsuz kare |
| 5. Araç Tespiti (COCO) | `OnnxVehicleDetector` | `ftr_main.py:172` | Kare | Araç kutuları + COCO kabin sınıfları |
| 6. Davranış Tespiti (Özel) | `OnnxCustomDetector` | `ftr_main.py:262` | Kare | 9 sınıf tespit listesi |
| 7. BoTSORT Takip | `SinapticTracker.update()` | `src/tracking_pipeline.py:330` | Tespitler + kare | Track ID'li tespitler |
| 8. Plaka OCR | `PlateRecognizer.recognize()` | `plate_ocr.py:122` | Plaka kırpması | Plaka metni + güven skoru |
| 9. Temporal CNN-LSTM | `ONNXTemporalClassifier.predict()` | `src/models/temporal/cnn_lstm.py` | 16 kare özellik dizisi | Davranış sınıfı + güven |
| 10. YOLO-LSTM Ensemble | `ftr_main.py:687` | `ftr_main.py` | YOLO + LSTM sonuçları | Ağırlıklı birleşik güven |
| 11. Olay Toplama | `CompetitionAdapter.observe_event()` | `src/competition_adapter.py:165` | Etiket + güven | Event segment |
| 12. Risk Skoru | `RiskScorer.compute()` | `risk_scorer.py:94` | Çok sinyalli faktörler | 0-100 skor |
| 13. QoD Kararı | `qod_orchestrator.py` | `api/` | Risk skoru + ağ metrikleri | CAMARA QoD isteği |
| 14. JSON Yazımı | `atomic_write_results()` | `src/competition_contract.py:107` | Doküman dict | `results.json` |

---

## 3. Sistem Katmanları

### 3.1 Veri Katmanı

| Özellik | Detay |
|---------|-------|
| **Amaç** | Ham video verisini analiz için hazırlamak |
| **Girdi** | MP4 video dosyası (yarışma girdisi) |
| **Çıktı** | Ön işlenmiş numpy frame dizisi |
| **Yöntem** | OpenCV VideoCapture + Zero-DCE parlaklık iyileştirme + Kamera kalibrasyon undistortion |
| **Dosyalar** | `ftr_main.py`, `camera_calibration.py`, `src/models/low_light/zero_dce.py` |
| **Güçlü Yönler** | Adaptif stride ile verimli örnekleme; koşullu Zero-DCE ile gece koşullarına dayanıklılık |
| **Sınırlılıklar** | Kamera kalibrasyon verisi dosya olmadığında tahminsel geri dönüşe geçilir; gerçek kalibrasyon katsayıları ölçülmemiştir |

### 3.2 Yapay Zekâ / Model Katmanı

| Özellik | Detay |
|---------|-------|
| **Amaç** | Nesne, davranış ve kimlik tespiti |
| **Girdi** | 640×640 normalize edilmiş kare tensörü |
| **Çıktı** | Bounding box, sınıf ID, güven skoru |
| **Yöntem** | YOLOv8m (ONNX) + CRNN/CTC (OCR) + CNN-LSTM (Temporal) |
| **Dosyalar** | `ftr_main.py`, `ai_pipeline.py`, `plate_ocr.py` |
| **Güçlü Yönler** | ONNX Runtime ile CPU/GPU sağlayıcı otomatik seçimi; warmup inference ile gecikme azaltımı |
| **Sınırlılıklar** | CPU'da ortalama single-frame 118ms gecikme; tam video için 15.2 saniye (8 saniyelik hedefin üzerinde) |

### 3.3 Video İşleme Katmanı

| Özellik | Detay |
|---------|-------|
| **Amaç** | Çerçeve bazlı paralel işleme ve asenkron video decode |
| **Girdi** | Ham video akışı |
| **Çıktı** | İşlenmiş kare + zaman damgası |
| **Yöntem** | `asyncio + ThreadPoolExecutor(max_workers=2)` ile non-blocking decode |
| **Dosyalar** | `ftr_main.py:405-729` |
| **Güçlü Yönler** | Asenkron I/O ile CPU/GPU çakışma süresi artırılır |
| **Sınırlılıklar** | 2 worker thread limitli; büyük videolarda bottleneck olabilir |

### 3.4 Takip / Tracker Katmanı

| Özellik | Detay |
|---------|-------|
| **Amaç** | Nesne kimliği tutarlılığını çok kare boyunca korumak |
| **Girdi** | Tespitler + BEV koordinatları |
| **Çıktı** | Track ID'li tespit listesi |
| **Yöntem** | BoTSORT (İki aşamalı Macar eşleştirme + Kalman filtresi + ReID) |
| **Dosyalar** | `src/tracking_pipeline.py` |
| **Güçlü Yönler** | BEV perspektif dönüşümü ile gerçek dünya mesafe tahmini; Kamera Hareket Telafisi (CMC) ile hareketli kamera desteği |
| **Sınırlılıklar** | ReID modeli (osnet_x0_25.onnx) mevcut değilse HSV histogram geri dönüşüne geçilir |

### 3.5 OCR / Plaka Katmanı

| Özellik | Detay |
|---------|-------|
| **Amaç** | Türk plakasını karakterlere dönüştürmek |
| **Girdi** | Plaka kırpma görüntüsü |
| **Çıktı** | Plaka metni (ör: `34TR1001`) + güven skoru |
| **Yöntem** | LPRNet (bbox tespiti) → CRNN + CTC Greedy Decoding → Confidence-weighted temporal voting |
| **Dosyalar** | `plate_ocr.py` |
| **Güçlü Yönler** | Edit distance tabanlı gruplama ile OCR gürültüsüne karşı dirençli temporal oylama |
| **Sınırlılıklar** | Test seti yalnızca 14 sentetik görüntüden oluşmakta; gerçek plaka veri seti ile doğrulama eksik |

### 3.6 Risk / Karar Katmanı

| Özellik | Detay |
|---------|-------|
| **Amaç** | Çok sinyalli risk skoru üretmek |
| **Girdi** | Hız, telefon, uyuklama, sigara, plaka tespit durumu |
| **Çıktı** | 0-100 arası normalize risk skoru + seviye (DÜŞÜK/ORTA/KRİTİK) |
| **Yöntem** | Ağırlıklı doğrusal toplama → normalize → eşik sınıflandırma |
| **Dosyalar** | `risk_scorer.py` |
| **Güçlü Yönler** | Track ID bazlı bağışlama mekanizması; bellek sızıntısına karşı TTL temizleme |
| **Sınırlılıklar** | Ağırlıklar deneysel, gerçek kaza verileriyle kalibre edilmemiş |

### 3.7 5G / MEC / API Katmanı

| Özellik | Detay |
|---------|-------|
| **Amaç** | Ağ kalitesini akıllıca yönetmek |
| **Girdi** | Risk skoru + ağ koşulları |
| **Çıktı** | CAMARA QoD oturum isteği / reddi |
| **Yöntem** | `CalibratedBenefitModel` → durum makinesi (IDLE→OBSERVE→REQUESTING→ACTIVE→COOLDOWN) |
| **Dosyalar** | `api/`, QoD orchestrator |
| **Güçlü Yönler** | Fayda güdümlü karar kapısı gereksiz QoD isteklerini engeller; cooldown koruması |
| **Sınırlılıklar** | Gerçek Turkcell CAMARA SIM entegrasyonu yapılamamış; yalnızca simülasyon doğrulaması mevcut |

### 3.8 Çıktı / Raporlama Katmanı

| Özellik | Detay |
|---------|-------|
| **Amaç** | Yarışma şemasına uygun JSON üretmek |
| **Girdi** | Toplanmış olay ve araç verileri |
| **Çıktı** | Atomik olarak yazılan `results.json` |
| **Yöntem** | JSON Schema Draft 2020-12 doğrulaması + `os.fsync()` ile atomik yazım |
| **Dosyalar** | `src/competition_contract.py`, `src/competition_adapter.py` |
| **Güçlü Yönler** | Şema ihlali varsa çıktı yazılmaz; `unknown` üretilmez; iç etiketler otomatik normalize edilir |
| **Sınırlılıklar** | Yalnızca tek araç (primary track) raporu; çok araçlı senaryolarda kısmi bilgi kaybı olabilir |

### 3.9 Test / Doğrulama Katmanı

| Özellik | Detay |
|---------|-------|
| **Amaç** | Sistem bütünlüğünü otomatik doğrulamak |
| **Girdi** | Smoke test videosu, birim testler |
| **Çıktı** | pytest raporu + JSON çıktısı |
| **Yöntem** | 79 pytest testi (birim + entegrasyon) + offline FTR kabul testi |
| **Dosyalar** | `tests/`, `scripts/run_ftr_acceptance.py` |
| **Güçlü Yönler** | 79/79 test başarılı; JSON şema doğrulaması + model hash kilidi doğrulaması dahil |
| **Sınırlılıklar** | T4 GPU ve gerçek Android WebRTC testleri ortam bağımlılığı nedeniyle HEDEF durumunda |

---

## 4. İki Çalışma Modu

### Mod 1: Offline FTR (Docker)

```
Docker Container
├── /app/data/input/video.mp4 (RO mount)
├── /app/data/output/results.json (yazılır)
├── models/detector.onnx (kilitli, SHA-256 doğrulamalı)
└── python ftr_main.py (internet YOK, Redis YOK, Android YOK)
```

### Mod 2: Canlı 5G Pipeline

```
Android CameraX → WebRTC RTP/RTCP → GPU Media Service
                ↓
         BFF (JWT + SDP)
                ↓
         Redis (60sn SDP postbox)
                ↓
         QoD Orchestrator → CAMARA API
```

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
