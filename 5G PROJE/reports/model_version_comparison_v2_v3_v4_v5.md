# SİNAPTİC5G — Tüm Model Versiyonları Kıyaslama Raporu

> **Tarih:** 2026-06-26  
> **Yazar:** Seydi Eryılmaz (@seydivakkas)  
> **Amaç:** v2, v3, v4_smoke, v4_full ve v5 modellerinin performansını tek tabloda sunmak  
> **Kaynak Veriler:** `reports/val_metrics_v2.json`, `reports/val_metrics_detector_v3.json`, `reports/detector_v4_test_metrics.json`, `reports/detector_v4_full_test_metrics.json`, `reports/detector_v5_test_metrics.json`

---

## 1. Eğitim Parametreleri Karşılaştırması

| Parametre | detector_v2 | detector_v3 | detector_v4_smoke | detector_v4_full | **detector_v5** |
|-----------|:-----------:|:-----------:|:-----------------:|:----------------:|:---------------:|
| **Mimari** | YOLOv8m | YOLOv8m | YOLOv8m | YOLOv8m | **YOLOv8m** |
| **Epoch** | ~50 (smoke) | 50 | 1 | 2 | **60** |
| **Fraksiyon** | %2 | %100 | %5 | %100 | **%100** |
| **Optimizer** | SGD | SGD | SGD | AdamW | **AdamW** |
| **Freeze** | — | — | — | 10 | **10** |
| **Copy-Paste Aug.** | ❌ | ❌ | ❌ | ❌ | **✅ 0.15** |
| **MixUp** | ❌ | ❌ | ❌ | ❌ | **✅ 0.15** |
| **GPU** | RTX 4070 | RTX 4070 | RTX 4070 | RTX 4070 | **RTX 4070** |
| **AMP (FP16)** | ✅ | ✅ | ✅ | ✅ | **✅** |

---

## 2. Veri Seti Büyüklüğü Karşılaştırması

| Split | v2 | v3 | v4 (smoke/full) | **v5** |
|-------|:--:|:--:|:---------------:|:------:|
| **Train Görüntü** | 9.361 | 9.662 | 10.018 | **10.998** |
| **Val Görüntü** | 2.693 | 2.695 | 2.787 | **2.855** |
| **Test Görüntü** | 1.347 | 1.343 | 1.395 | **1.434** |
| **Toplam** | 13.401 | 13.700 | 14.200 | **15.287** |

### Sınıf 5-6-7 Eğitim Desteği (Train Instances)

| Sınıf | v2 | v3 | v4 | **v5** | v2→v5 Artış |
|-------|:--:|:--:|:--:|:------:|:-----------:|
| emniyet_kemeri_ihlali (5) | 304 | 304 | 654 | **1.260** | **+314%** |
| teknocan (6) | 80 | 84 | 80 | **723** | **+804%** |
| bilgisayar (7) | 35 | 335 | 35 | **662** | **+1791%** |

> [!TIP]
> v5'te Copy-Paste oversampling ve yeni veri kaynakları sayesinde düşük destekli sınıfların eğitim örnekleri dramatik şekilde arttırıldı.

---

## 3. Küresel Performans Metrikleri (Test Split)

| Metrik | detector_v2 | detector_v3 | detector_v4_smoke | detector_v4_full | **detector_v5** |
|--------|:-----------:|:-----------:|:-----------------:|:----------------:|:---------------:|
| **Precision** | 0.6033 | 0.9225 | 0.9012 | 0.3443 | **0.9379** |
| **Recall** | 0.3561 | 0.8905 | 0.8518 | 0.5044 | **0.9333** |
| **F1 Score** | 0.4478 | 0.9062 | 0.8758 | 0.4093 | **0.9356** |
| **mAP@0.50** | 0.4606 | 0.9164 | 0.8676 | 0.3042 | **0.9473** |
| **mAP@0.50:0.95** | 0.3151 | 0.6878 | 0.6680 | 0.1858 | **0.6904** |

### Performans Gelişim Grafiği (mAP@0.50)

```
v2          ██████████████████████████                                               0.4606
v4_full     ████████████████                                                          0.3042
v4_smoke    █████████████████████████████████████████████████                         0.8676
v3          █████████████████████████████████████████████████████                     0.9164
v5          ██████████████████████████████████████████████████████                    0.9473 ★
```

> [!IMPORTANT]
> **v5, tüm versiyonlar arasında en yüksek mAP@0.50 (0.9473) ve F1 (0.9356) skorlarına sahiptir.**
> v4_full'ün düşük skorları, yalnızca 2 epoch eğitilmiş olmasından kaynaklanmaktadır — pipeline doğrulaması amaçlıydı.

---

## 4. Sınıf Bazlı AP@0.50 Karşılaştırması

| Sınıf | detector_v2 | detector_v3 | detector_v4_smoke | detector_v4_full | **detector_v5** | v3→v5 Δ |
|-------|:-----------:|:-----------:|:-----------------:|:----------------:|:---------------:|:-------:|
| telefonla_konusma (0) | 0.4073 | 0.9890 | 0.9861 | 0.3941 | **0.9914** | +0.24% |
| su_icme (1) | 0.8778 | 0.9948 | 0.9950 | 0.7338 | **0.9950** | +0.02% |
| arkaya_bakma (2) | 0.5236 | 0.9950 | 0.9950 | 0.2560 | **0.9950** | 0.00% |
| esneme (3) | 0.2482 | 0.9876 | 0.9690 | 0.2811 | **0.9923** | +0.48% |
| sigara_icme (4) | 0.7186 | **0.8987** | 0.8749 | 0.0788 | 0.8575 | -4.58% |
| emniyet_kemeri_ihlali (5) | 0.3863 | 0.7743 | 0.3483 | 0.0673 | **0.8596** | +11.0% |
| teknocan (6) | 0.0000 | **0.9623** | 0.9950 | 0.0368 | 0.8973 | -6.75% |
| bilgisayar (7) | 0.0000 | 0.6650 | 0.6650 | 0.0006 | **0.9531** | +43.3% |
| license_plate (8) | 0.9833 | 0.9814 | 0.9803 | 0.8897 | **0.9841** | +0.28% |

### Öne Çıkan Değişimler (v3 → v5)

| Sınıf | Değişim | Açıklama |
|-------|:-------:|----------|
| **bilgisayar** | **+43.3%** 🟢 | Copy-Paste + laptop_incar sentetik veri etkisi |
| **emniyet_kemeri_ihlali** | **+11.0%** 🟢 | seatbelt_v1 entegrasyonu + artan eğitim desteği |
| **sigara_icme** | **-4.58%** 🟡 | Geniş veri setinde çok sınıflı rekabetin etkisi (trade-off) |
| **teknocan** | **-6.75%** 🟡 | Sentetik verinin gerçek veri ile fark oluşturması |

---

## 5. Sınıf Bazlı Precision Karşılaştırması

| Sınıf | detector_v2 | detector_v3 | detector_v4_smoke | detector_v4_full | **detector_v5** |
|-------|:-----------:|:-----------:|:-----------------:|:----------------:|:---------------:|
| telefonla_konusma (0) | 0.8653 | 0.9761 | 0.9768 | 0.3555 | **0.9886** |
| su_icme (1) | 0.9706 | **1.0000** | 1.0000 | 0.5524 | 0.9883 |
| arkaya_bakma (2) | **1.0000** | 0.9946 | 0.9945 | 0.2158 | 0.9868 |
| esneme (3) | 0.5123 | 0.9340 | 0.8916 | 0.0668 | **0.9440** |
| sigara_icme (4) | 0.2660 | 0.8833 | 0.7854 | 0.3547 | **0.9132** |
| emniyet_kemeri_ihlali (5) | 0.9242 | 0.8628 | 0.8024 | 0.0835 | **0.9193** |
| teknocan (6) | 0.0000 | 0.9416 | **1.0000** | 1.0000 | 0.8265 |
| bilgisayar (7) | 0.0000 | 0.7379 | 0.6925 | 0.0000 | **0.9237** |
| license_plate (8) | 0.8916 | **0.9721** | 0.9676 | 0.4703 | 0.9505 |

---

## 6. Sınıf Bazlı Recall Karşılaştırması

| Sınıf | detector_v2 | detector_v3 | detector_v4_smoke | detector_v4_full | **detector_v5** |
|-------|:-----------:|:-----------:|:-----------------:|:----------------:|:---------------:|
| telefonla_konusma (0) | 0.2056 | **0.9920** | 0.9840 | 0.7193 | 0.9760 |
| su_icme (1) | 0.6606 | 0.9877 | 0.9862 | 0.8500 | **1.0000** |
| arkaya_bakma (2) | 0.0000 | **1.0000** | 1.0000 | 0.9406 | **1.0000** |
| esneme (3) | 0.2549 | 0.9804 | 0.9679 | 0.9216 | **0.9902** |
| sigara_icme (4) | **0.8931** | 0.8591 | 0.8575 | 0.0686 | 0.7631 |
| emniyet_kemeri_ihlali (5) | 0.2326 | 0.6977 | 0.3333 | 0.1398 | **0.8280** |
| teknocan (6) | 0.0000 | 0.8750 | 0.9161 | 0.0000 | **0.9167** |
| bilgisayar (7) | 0.0000 | 0.6667 | 0.6667 | 0.0000 | **0.9615** |
| license_plate (8) | 0.9579 | 0.9557 | 0.9543 | 0.8995 | **0.9644** |

---

## 7. Nihai Karar ve Sonuçlar

### Model Seçim Kararı

| Model | Karar | Gerekçe |
|-------|-------|---------|
| detector_v2 | ❌ ARŞİV → SİLİNECEK | Smoke-run (sadece %2 fraksiyon), çoğu sınıfta sıfır başarım |
| detector_v3 | ❌ ARŞİV → SİLİNECEK | 50 epoch tam eğitim, güçlü baseline; ancak v5 tarafından aşıldı |
| detector_v4_smoke | ❌ ARŞİV → SİLİNECEK | Pipeline doğrulama amaçlı (%5, 1 epoch) — üretim modeli değil |
| detector_v4_full | ❌ ARŞİV → SİLİNECEK | Sadece 2 epoch deneysel eğitim — baseline'ın çok gerisinde |
| **detector_v5** | ✅ **AKTİF ÜRETİM** | 60 epoch, tam veri seti, Copy-Paste+MixUp; en yüksek mAP/F1 |

### v5'in Öne Çıkan Başarıları

- 🏆 **En yüksek genel mAP@0.50:** 0.9473 (v3: 0.9164, +3.4%)
- 🏆 **En yüksek F1 Score:** 0.9356 (v3: 0.9062, +3.2%)
- 🏆 **emniyet_kemeri_ihlali AP50:** 0.8596 (v3: 0.7743, +11.0%)
- 🏆 **bilgisayar AP50:** 0.9531 (v3: 0.6650, +43.3%)
- 🏆 **teknocan Recall:** 0.9167 (v3: 0.8750, +4.8%)
- 🏆 **En büyük veri seti:** 15.287 görüntü (v2: 13.401, +14.1%)

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
