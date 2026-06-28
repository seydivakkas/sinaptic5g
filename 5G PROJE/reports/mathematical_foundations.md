# SİNAPTİC5G — Matematiksel ve Algoritmik Dayanaklar

> **Sürüm:** 1.0  
> **Tarih:** 2026-06-25  
> **Yazar:** Seydi Eryılmaz (@seydivakkas)

---

## 1. YOLO Nesne Tespiti

### 1.1 Bounding Box Temsili

YOLO ağları her tahmin için bir bounding box'ı merkez koordinatları ve boyut cinsinden tahmin eder:

```
Tahmin vektörü: [cx, cy, bw, bh, c_obj, c_0, c_1, ..., c_(K-1)]
```

- `(cx, cy)`: Kutu merkezi (hücre içinde göreceli konum)
- `(bw, bh)`: Tahmin edilen genişlik ve yükseklik
- `c_obj`: Nesne güven puanı
- `c_k`: k. sınıf puanı

Piksel koordinatlarına dönüşüm (bu projede `ftr_main.py:232`):

```python
x1 = max(0, int((cx - bw/2) * scale_x))
y1 = max(0, int((cy - bh/2) * scale_y))
x2 = min(W, int((cx + bw/2) * scale_x))
y2 = min(H, int((cy + bh/2) * scale_y))
```

**Bu projede kullanımı:** `OnnxCustomDetector.detect()` ve `OnnxVehicleDetector.detect()` fonksiyonları YOLOv8 ONNX çıktısını bu formülle orijinal kare boyutuna ölçeklendirir.

**Teknik açıklama:** YOLOv8m modeli 640×640 normalize girdi üzerinde çalışır; model çıktısı [1, 4+C, 8400] formatında 8400 tahmin hücresi içerir. Her hücre için argmax ile en yüksek sınıf puanı seçilir, güven eşiğini geçen tespitler NMS'e aktarılır.

---

### 1.2 Non-Maximum Suppression (NMS)

Örtüşen bounding box'ları elemek için NMS uygulanır:

**Algoritma:**
1. Tespitleri güven skoruna göre azalan sırada sırala
2. En yüksek skorlu kutuyu seç (S)
3. Kalan kutularla IoU hesapla
4. IoU > eşik olan kutuları liste dışına al
5. Kalan kutular için 2-4 adımlarını tekrarla

**Bu projede kullanımı:** `cv2.dnn.NMSBoxes(boxes, scores, conf_threshold=0.35, iou_threshold=0.45)`

**Teknik açıklama:** Güven eşiği 0.35, IoU eşiği 0.45 olarak ayarlanmıştır. Bu parametreler, düşük destekli sınıflarda yeterli recall sağlarken yüksek FP oranını kontrol altında tutar.

---

## 2. IoU (Intersection over Union)

IoU iki bounding box arasındaki örtüşme oranını ölçer:

```
IoU(A, B) = |A ∩ B| / |A ∪ B| = kesişim_alanı / (alan_A + alan_B - kesişim_alanı)
```

**Bu projede kullanımı:** `SinapticTracker._iou()` (tracking_pipeline.py:362):

```python
inter = max(0, xB - xA) * max(0, yB - yA)
union = areaA + areaB - inter
return inter / (union + 1e-6)
```

**Teknik açıklama:** Epsilon (1e-6) ile sıfıra bölme önlenir. BoTSORT takip motorunda maliyet matrisinin birinci bileşeni olarak IoU kullanılır (w₁=0.5).

---

## 3. Precision, Recall, F1, AP, mAP

### 3.1 Temel Metrikler

```
Precision (P) = TP / (TP + FP)
Recall (R)    = TP / (TP + FN)
F1            = 2 × P × R / (P + R)
```

**Değişken tanımları:**
- TP: Doğru pozitif (gerçek nesne IoU≥0.50 ile eşleşti)
- FP: Yanlış pozitif (eşleşen gerçek yok veya IoU<0.50)
- FN: Yanlış negatif (kaçırılan gerçek nesne)

### 3.2 Average Precision (AP)

AP, precision-recall eğrisinin altındaki alanı hesaplar:

```
AP = ∫₀¹ P(r) dr  ≈  Σₖ [R(k) - R(k-1)] × P(k)
```

### 3.3 mAP@0.50 ve mAP@0.50:0.95

```
mAP@0.50    = (1/K) × Σ_k AP_k(IoU=0.50)
mAP@0.50:95 = (1/10) × Σ_{t=0.50,0.55,...,0.95} mAP@t
```

**Bu projede detector_v5 test sonuçları:**

| Metrik | Değer | Kaynak |
|--------|-------|--------|
| mAP@0.50 | **0.9334** | `reports/detector_v5_test_metrics.json` |
| mAP@0.50:0.95 | **0.6687** | `reports/detector_v5_test_metrics.json` |
| Precision | **0.9289** | `reports/detector_v5_test_metrics.json` |
| Recall | **0.9169** | `reports/detector_v5_test_metrics.json` |
| F1 | **0.9229** | `reports/detector_v5_test_metrics.json` |

**Teknik açıklama:** mAP@0.50 metriği, her sınıf için eşik IoU=0.50 koşulunu sağlayan tespitler üzerinden hesaplanan ortalama hassasiyeti temsil eder. detector_v5'in 0.9334 mAP@0.50 değeri, önceki üretim modeli detector_v3'ün 0.9164 değerini %1.8 oranında aşmaktadır.

---

## 4. Kalman Filtresi (Takip)

`AdaptiveKalmanTracker` (tracking_pipeline.py:209), BEV koordinatlarında 4 boyutlu durum vektörü kullanır:

```
Durum vektörü: x = [bev_x, bev_y, vx, vy]ᵀ
Ölçüm vektörü: z = [bev_x, bev_y]ᵀ
```

**Geçiş matrisi (F) — sabit hız modeli:**
```
F = [[1, 0, dt, 0 ],
     [0, 1, 0,  dt],
     [0, 0, 1,  0 ],
     [0, 0, 0,  1 ]]
```

**Ölçüm matrisi (H):**
```
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

**Tahmin adımı:**
```
x̂⁻ₖ = F × x̂ₖ₋₁
P⁻ₖ  = F × Pₖ₋₁ × Fᵀ + Q
```

**Güncelleme adımı:**
```
Kₖ   = P⁻ₖ × Hᵀ × (H × P⁻ₖ × Hᵀ + R)⁻¹
x̂ₖ  = x̂⁻ₖ + Kₖ × (zₖ - H × x̂⁻ₖ)
Pₖ   = (I - Kₖ × H) × P⁻ₖ
```

**Adaptif ölçüm gürültüsü:**
```python
noise_factor = 1.0 + 3.0 × (1.0 - max(confidence, 0.30))
R_val = base_R × noise_factor
```

**Teknik açıklama:** Kalman filtresi tespit güven skoruna bağlı olarak ölçüm gürültü kovaryans matrisini (R) dinamik olarak günceller. Düşük güvenli tespitler daha büyük R değeriyle daha az ağırlık alır; bu sayede gürültülü YOLO çıktılarında takip tutarlılığı korunur.

---

## 5. Hungarian (Macar) Algoritması

`SinapticTracker._hungarian_match()` (tracking_pipeline.py:424) scipy `linear_sum_assignment` kullanır:

**Maliyet matrisi oluşturumu:**
```
cost(i,j) = 0.5 × IoU(track_i, det_j) + 0.3 × BEV_proximity(i,j) + 0.2 × ReID_sim(i,j)
```

**Optimizasyon:**
```
argmin Σᵢ cost[row_ind[i], col_ind[i]]
```

Scipy'nin `linear_sum_assignment` fonksiyonu O(n³) karmaşıklıklı Macar algoritmasını uygular.

**Geri dönüş:** Scipy yoksa `_scipy_lsa = False` kontrolüyle açgözlü (greedy) eşleştirme devreye girer.

**Teknik açıklama:** İki aşamalı eşleştirme stratejisi uygulanır: yüksek güvenli tespitler (≥0.45) birinci aşamada takiplerle eşleştirilir; kalan düşük güvenli tespitler (0.15–0.45) kayıp takiplerle ikinci aşamada eşleştirilmeye çalışılır.

---

## 6. CTC (Connectionist Temporal Classification) Dekodlama

CRNN modelinin çıktısı her zaman adımında sınıf logitlerini verir:

```
CRNN çıktısı: T × K  (T: zaman adımı, K: karakter sınıfı sayısı)
```

**Greedy CTC Dekodlama** (plate_ocr.py:231):

```python
char_indices = np.argmax(preds, axis=-1)  # T boyutunda argmax
decoded = []
prev_idx = -1
for idx in char_indices:
    if idx != 0 and idx != prev_idx:  # blank(0) ve tekrarlananları atla
        decoded.append(VOCABULARY[idx])
    prev_idx = idx
```

**Sözlük:** `"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"` (37 karakter, indeks 0 = blank)

**Teknik açıklama:** CTC kaybı, model çıktısı ile hedef dizi uzunlukları farklı olduğunda bile eğitimi mümkün kılan bir kayıp fonksiyonudur. Greedy dekodlama, her zaman adımında en yüksek olasılıklı karakteri seçer ve blank token ile ardışık tekrarları kaldırır; bu yöntem olası tüm dizileri değerlendiren beam search'e kıyasla daha hızlı fakat daha düşük doğrulukludur.

---

## 7. Risk Skoru Formülü

`RiskScorer.compute()` (risk_scorer.py:94) ağırlıklı doğrusal toplama kullanır:

```
raw_score = 35 × speed_factor(excess_kmh)
          + 30 × has_phone
          + 25 × is_drowsy
          + 15 × has_cigarette
          + 10 × (not plate_detected AND NOT grace_period)
          +  8 × is_yawning
          + 10 × has_toy_on_wheel

final_score = min(100.0, raw_score)
```

**Hız faktörü:**
```
speed_factor = min(excess_kmh / 20.0, 1.0)   # excess = speed - speed_limit
```

**Risk seviyeleri:**
```
DÜŞÜK  : score < 30
ORTA   : 30 ≤ score < 70
KRİTİK : score ≥ 70
```

**QoD tetikleme eşiği:** `needs_qod = score ≥ 45`

**Teknik açıklama:** Maksimum teorik ham puan 133'tür (tüm faktörler aynı anda etkin); bu değer 100'e normalizasyon ile örtük olarak sağlanır. Hız bileşeni doğrusal ve sınırlandırılmış (0–35 puan) olup 20 km/s üzerindeki hız artışları tahrik mekanizmasını tam kapasitede çalıştırır.

---

## 8. Veri Seti Bölme Stratejisi

### 8.1 Split Oranları

`configs/data.yaml`:
```yaml
split:
  policy: group_aware
  ratios: [0.70, 0.15, 0.15]
  seed: 42
  group_keys: [video_id, capture_session, vehicle_id, person_id]
```

**detector_v5 Gerçek Sonuçları:**

| Split | Görüntü | Oran |
|-------|---------|------|
| Train | 10,998 | ~72.4% |
| Val | 2,855 | ~18.8% |
| Test | 1,395 | ~9.2% |
| **Toplam** | **15,248** | 100% |

### 8.2 Veri Sızıntısı Kontrolü

Group-aware split (`group_keys`), aynı video oturumuna ait karelerin farklı split'lere dağılmasını önler. Bu, değerlendirme metriklerinin gerçek genelleme yeteneğini doğru yansıtması için kritiktir. `detector_v4_pretrain_leakage_check.json` dosyasında leakage=false kaydı mevcuttur.

### 8.3 Sınıf Dengeleme (Oversampling)

Düşük destekli sınıflar için Copy-Paste augmentasyon ile örnekler çoğaltılmıştır:

| Sınıf | Train Öncesi | Train Sonrası | Artış |
|-------|:---:|:---:|:---:|
| teknocan (ID 6) | 90 | 723 | +703% |
| emniyet_kemeri (ID 5) | 654 | 1,260 | +93% |
| bilgisayar (ID 7) | 335 | 662 | +98% |

**Teknik açıklama:** Sınıf imbalance, nadir sınıflar için yüksek yanlış negatif oranına yol açar. Copy-Paste augmentasyon, düşük temsilli sınıf örneklerini rastgele arka planlara yapıştırarak sentetik eğitim verisi üretir; bu yöntem, nadir sınıf tespitinde Focal Loss'a alternatif ya da tamamlayıcı bir strateji olarak etkin biçimde çalışmaktadır.

---

## 9. Özellik Çıkarımı — Temporal CNN-LSTM Giriş Vektörü

7 boyutlu özellik vektörü her kare için hesaplanır:

```python
feat = [
    EAR,                           # Göz kırpma oranı (0.0 - 0.5)
    MAR,                           # Ağız açıklık oranı (0.0 - 1.0)
    speed_kmh / 120.0,             # Normalize hız (0.0 - 1.0)
    head_angle_norm,               # Kafa açısı normalize (MediaPipe)
    float(has_phone),              # 0/1 telefon kullanım bayrağı
    float(has_cigarette),          # 0/1 sigara bayrağı
    max(0.0, (speed - 50) / 50.0) # Hız aşımı skoru
]
```

16 ardışık kare biriktirilir → `[1, 16, 7]` tensör → CNN-LSTM sınıflandırıcıya beslenir.

**Teknik açıklama:** Giriş özellik vektörünün hem eşzamanlı (YOLO tespiti, kafa pozisyonu) hem de zaman-bağımlı (hız geçmişi) sinyalleri birleştirmesi, CNN-LSTM mimarisinin kısa süreli belleğini (LSTM gizli durum) sürücü davranışının geçici örüntülerini öğrenmek için etkin biçimde kullanmasını sağlamaktadır.

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
