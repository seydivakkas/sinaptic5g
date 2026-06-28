# SİNAPTİC5G — Yapay Zekâ Çözümü ve Yöntemleri Raporu

> **Sürüm:** 1.0  
> **Tarih:** 2026-06-25  
> **Yazar:** Seydi Eryılmaz (@seydivakkas)

---

## 1. Kullanılan Yapay Zekâ Modelleri

### 1.1 detector_v5 — Ana Nesne/Davranış Dedektörü (Üretim)

| Özellik | Değer |
|---------|-------|
| **Görev** | Sürücü davranışı ve nesne tespiti (9 sınıf) |
| **Temel Model** | YOLOv8m (Ultralytics 8.4.41) |
| **Format** | ONNX (FP16, opset=17, simplify=True) |
| **Boyut** | ~49.47 MB (ONNX) |
| **Girdi** | [1, 3, 640, 640] float32 tensörü |
| **Çıktı** | [1, 13, 8400] bounding box + sınıf skoru tensörü |
| **Eğitim** | 60 epoch, RTX 4070 Laptop (8GB VRAM), AdamW, batch=12, AMP=FP16 |
| **Üretim Durumu** | Aktif üretim modeli (models/detector.onnx + detector_optimized.onnx) |
| **SHA-256 (ONNX)** | `da1840434b6a13eae5205ff5ce7e41c60edc52d93a11c0d422fdaa86a966d5b9` |
| **SHA-256 (best.pt)** | `69f579ad3e429542b64ecdeafbe7bd027d9ed8309d84be8f2775d00b7f484778` |

**Tespit Edilen 9 Sınıf:**

| ID | Sınıf Adı | Test Desteği | Test AP50 | Test Recall |
|----|-----------|:---:|:---:|:---:|
| 0 | telefonla_konusma | 250 | 0.9914 | 0.9760 |
| 1 | su_icme | 100 | 0.9950 | 1.0000 |
| 2 | arkaya_bakma | 101 | 0.9950 | 1.0000 |
| 3 | esneme | 102 | 0.9923 | 0.9895 |
| 4 | sigara_icme | 758 | 0.8577 | 0.7614 |
| 5 | emniyet_kemeri_ihlali | 93 | 0.8595 | 0.8280 |
| 6 | teknocan | 10 | 0.9783 | 0.9000 |
| 7 | bilgisayar | 6 | 0.7517 | 0.8333 |
| 8 | license_plate | 219 | 0.9795 | 0.9639 |

**Avantajlar:**
- YOLOv8m transfer öğrenimi ile hızlı yakınsama
- FP16 AMP ile GPU hızlandırma
- Copy-paste augmentasyon ile düşük destekli sınıfların başarıyla öğrenilmesi
- Tek geçişte hem davranış hem plaka tespiti

**Dezavantajlar:**
- CPU'da ~118ms/frame; gerçek zamanlı için GPU zorunlu
- `sigara_icme` (ID 4) ve `emniyet_kemeri_ihlali` (ID 5) sınıf imbalance nedeniyle görece düşük recall
- Test split'inde `teknocan` için yalnızca 10 örnek; istatistiksel güvenilirlik sınırlı

---

### 1.2 OnnxVehicleDetector — COCO Araç Dedektörü

| Özellik | Değer |
|---------|-------|
| **Görev** | Genel araç tespiti (sedan, minibus, kamyon) |
| **Model** | YOLOv8n (COCO eğitimli, 80 sınıf) |
| **Format** | ONNX (`yolov8n.onnx`) |
| **Girdi** | [1, 3, 640, 640] |
| **Çıktı** | Bounding box + COCO sınıf ID |
| **Kullanım** | COCO class 2 (car→sedan), 5 (bus→minibus), 7 (truck→kamyon) |
| **Kabin Sınıfları** | COCO 0 (person), 73 (laptop) → F6 kabin ROI için |
| **Üretim Durumu** | Aktif (canlı ve FTR pipeline) |

---

### 1.3 LPRNet — Plaka Bölge Dedektörü

| Özellik | Değer |
|---------|-------|
| **Görev** | Görüntü içinde plaka bölgesini bounding box ile saptamak |
| **Format** | ONNX (`models/lprnet.onnx`, ~1.79 MB) |
| **Girdi** | [1, 3, 96, 320] (320×96 piksel) |
| **Çıktı** | [1, 4] normalize koordinat (x1, y1, x2, y2) |
| **SHA-256** | `6c8ff0a71bc4bb0d7c5a8ee87a2c77b3df0a99a0122764d18c6260875383aa50` |
| **Üretim Durumu** | Aktif (plate_ocr.py, OCR ilk aşama) |

---

### 1.4 CRNN + CTC — Karakter Tanıma Modeli

| Özellik | Değer |
|---------|-------|
| **Görev** | Plaka görüntüsündeki karakterleri okumak |
| **Model** | Convolutional Recurrent Neural Network (CRNN) |
| **Format** | ONNX (`models/crnn.onnx`, ~44.9 MB) |
| **Girdi** | [1, 1, 32, 160] gri tonlamalı, CLAHE iyileştirilmiş |
| **Çıktı** | [seq_len, batch, num_classes] karakter logit tensörü |
| **Dekodlama** | CTC Greedy Decoding (blank karakter = indeks 0) |
| **Sözdizimi** | 37 karakter: `-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` |
| **SHA-256** | `f95f0b8d5f23d846ed33491002c130bd1fdba18c7e3c5dd773c55dec9ce3cc53` |
| **Üretim Durumu** | Aktif (plate_ocr.py, OCR ikinci aşama) |
| **Ayrıca** | `crnn_int8.onnx` INT8 nicemleme sürümü mevcut (~11.8 MB) |

---

### 1.5 CNN-LSTM — Temporal Davranış Sınıflandırıcı

| Özellik | Değer |
|---------|-------|
| **Görev** | 16 ardışık kareden oluşan özellik dizisinden davranış sınıfı tahmini |
| **Format** | ONNX (`models/cnn_lstm.onnx`, ~2.4 KB — stub/hedef) |
| **Girdi** | [1, 16, 7] — 16 kare × 7 özellik vektörü |
| **7 Özellik** | EAR, MAR, speed_norm, head_angle_norm, has_phone, has_cigarette, speed_excess |
| **Çıktı** | Davranış sınıfı + güven skoru |
| **SHA-256** | `c432cefec4686699fec7271c5260b22f167e61c504741723e3794863f09f3039` |
| **Üretim Durumu** | HEDEF — `lstm_status: "HEDEF"` (model_lock.json'da kayıtlı) |
| **Kullanım** | `ftr_main.py --enable-lstm` bayrağı ile etkinleştirilir |

---

### 1.6 Zero-DCE — Düşük Işık İyileştirme

| Özellik | Değer |
|---------|-------|
| **Görev** | Düşük parlaklıklı kareleri iyileştirmek |
| **Tetikleme** | Ortalama parlaklık < 80 (ZERO_DCE_BRIGHTNESS_THRESHOLD) |
| **Dosya** | `src/models/low_light/zero_dce.py` |
| **Üretim Durumu** | Aktif (koşullu çalıştırma) |
| **Avantaj** | Gece koşullarında tespit kalitesi artışı |
| **Dezavantaj** | Ek işlem yükü (CPU'da ms cinsinden); düşük ışık dışında devre dışı |

---

### 1.7 BoTSORTReID — Yeniden Tanımlama (ReID) Modeli

| Özellik | Değer |
|---------|-------|
| **Görev** | Aynı nesnenin farklı karelerdeki kimlik tutarlılığı |
| **Tercih Edilen** | OSNet-x0.25 ONNX (`models/osnet_x0_25.onnx`) |
| **Geri Dönüş** | HSV histogram imzası (model yoksa) |
| **Girdi** | [1, 3, 256, 128] normalize crop |
| **Çıktı** | 512-boyutlu cosine-benzerlik gömme vektörü |
| **Dosya** | `src/tracking_pipeline.py:117` |
| **Üretim Durumu** | Geri dönüş modunda aktif (osnet model opsiyonel) |

---

## 2. Özgün Yöntemler ve Katkılar

### 2.1 Adaptif Stride (ftr_main.py:81)

**Problem:** Sabit kare örnekleme hızı, CPU üzerinde aşırı yük veya kritik anlarda yetersiz örnekleme yaratır.

**Çözüm:** `compute_adaptive_stride()` fonksiyonu dört koşula göre stride'ı dinamik ayarlar:
- GPU varsa daha sık örnekleme (0.10–0.34s); GPU yoksa daha seyrek (0.35–1.0s)
- Tespit varsa stride küçültülür (kritik anlarda daha sık)
- Video sonuna yaklaşıldığında (<10% kalan) stride ≤0.5s ile sınırlanır

**Rapor Cümlesi:** Geliştirilen adaptif örnekleme stratejisi, hesaplama bütçesini sahnedeki anlık tehlike durumuna göre dinamik olarak yeniden dağıtmakta; böylece kritik tespit anlarında yüksek örnekleme sıklığı sağlanırken durağan sahnelerde işlem yükü düşürülmektedir.

### 2.2 Sınıf Bazlı Güven Eşikleri (src/class_registry.py:112)

**Problem:** Tek eşikli sistemlerde yüksek destekli sınıflar düşük destekli sınıfları maskeler.

**Çözüm:** `FTR_CLASS_THRESHOLDS` sözlüğü her sınıf için bağımsız güven eşiği tanımlar:
- `arkaya_bakma`, `esneme`: 0.25 (düşük precision, recall artırma)
- `sigara_icme`: 0.45 (yüksek FP → eşik yüksek)
- `emniyet_kemeri_ihlali`: 0.30 (güvenlik kritik → düşük eşik)

**Rapor Cümlesi:** Threshold kalibrasyon çalışmasına dayanan sınıf-spesifik güven eşik tablosu, her sınıfın precision-recall dengesini bağımsız olarak optimize etmekte; güvenlik-kritik davranışlar için hatırlatıcı değer düşürülürken yüksek yanlış pozitif üreten sınıflarda eşik yukarı alınmaktadır.

### 2.3 Temporal Voting + Edit Distance (plate_ocr.py:272)

**Problem:** Tek kare OCR sonuçları gürültüye karşı hassastır.

**Çözüm:** Son 7 kare OCR sonuçları tamponlanır. Edit distance ≤2 olan tahminler aynı kümeye alınır; küme içinde confidence-weighted oylama yapılır.

**Rapor Cümlesi:** Önerilen güven-ağırlıklı zamansal oylama ve Levenshtein mesafesi tabanlı gruplama yöntemi, tek kare OCR hatalarını baskılayarak çok-kare üzerinde tutarlı plaka tahminleri elde etmekte; bu sayede gerçek dünya gürültüsüne karşı yüksek doğruluk korunmaktadır.

### 2.4 BoTSORT BEV + Kalman (src/tracking_pipeline.py)

**Problem:** Piksel koordinatlı takip, kamera açısı değiştiğinde bozulur; gerçek hız hesaplanamaz.

**Çözüm:** Kuşbakışı (BEV - Bird's Eye View) perspektif dönüşümü ile piksel koordinatları metre cinsinden gerçek dünya koordinatlarına çevrilir. Kalman filtresi bu BEV koordinatları üzerinde çalışır.

**Rapor Cümlesi:** Kamera perspektif homografisi ile elde edilen kuşbakışı koordinat dönüşümü, Kalman filtresi tabanlı çok nesne takibini gerçek dünya metrik uzayında çalıştırmakta; böylece hem tutarlı nesne kimliği hem de fiziksel anlamlı hız tahmini aynı çerçevede mümkün kılınmaktadır.

### 2.5 YOLO-LSTM Ensemble (ftr_main.py:686)

**Problem:** YOLO tek kare tespiti ile LSTM zaman serisi sınıflandırması bağımsız çalışırsa çakışma üretir.

**Çözüm:** İki modelin çıktısı ağırlıklı ortalamayla birleştirilir: `ensemble_conf = 0.70 × YOLO_conf + 0.30 × LSTM_conf`. YOLO tespit etmediği ama LSTM tespit ettiği olaylar 0.30× faktörüyle eklenir.

**Rapor Cümlesi:** Kare-bazlı YOLO dedektörü ve zaman-serisi CNN-LSTM sınıflandırıcısının ağırlıklı ensemble bütünleşmesi, anlık ve geçici davranış sinyallerini bir araya getirerek hem hızlı hem de kalıcı tehlike tespitini aynı karar çerçevesinde ele almaktadır.

### 2.6 Üretim Model Kilitleme (model_lock.json)

**Problem:** Model güncellemeleri sırasında onaylanmamış modelin sisteme girmesi riski.

**Çözüm:** `verify_model_lock()` başlangıçta çalışır; SHA-256 bütünlük kontrolü başarısız olursa sistem hatayla kapanır (fail-closed). Yeni model yalnızca açık kabul kapısı (teknocan Recall ≥0.30, mAP50 ≥ baseline) geçtikten sonra promote edilir.

**Rapor Cümlesi:** Kriptografik hash kilitleme ve açık kabul kapısı kriterleri aracılığıyla uygulanan üretim model güvenlik süreci, onaylanmamış model sürümlerinin sisteme girmesini önlemekte ve her model güncellemesini ölçülebilir performans kanıtına dayandırmaktadır.

### 2.7 Offline FTR + Docker Modu

**Problem:** Yarışma ortamı internet bağlantısı veya harici servis gerektirmeyebilir.

**Çözüm:** `ftr_main.py` tamamen çevrimdışı çalışır: `--network none` Docker flag ile derlenir; çıktı `os.fsync()` ile atomik yazılır; şema doğrulaması başarısız olursa dosya hiç yazılmaz.

**Rapor Cümlesi:** Geliştirilen çevrimdışı Docker gönderim hattı, ağ bağlantısı veya dış servis gerektirmeksizin atomik ve şema-doğrulamalı sonuç üretmekte; böylece yarışma değerlendirme ortamına tam uyumluluk sağlanmaktadır.

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
