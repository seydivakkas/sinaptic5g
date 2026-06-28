# SİNAPTİC5G — Veri Seti Denetim Raporu

> **Sürüm:** 1.0  
> **Tarih:** 2026-06-25  
> **Yazar:** Seydi Eryılmaz (@seydivakkas)  
> **Kanıt Kaynakları:** `reports/detector_v5_split_summary.json`, `configs/data_v5.yaml`, `reports/detector_v5_class_comparison.csv`

---

## 1. Veri Seti Genel Özeti

| Parametre | Değer | Kaynak |
|-----------|-------|--------|
| **Toplam Görüntü** | 15,248 | detector_v5_split_summary.json |
| **Sınıf Sayısı (FTR modeli)** | 9 | FTR_CLASS_MAP (class_registry.py) |
| **Sınıf Sayısı (Canlı model)** | 6 | LIVE_CLASS_MAP (class_registry.py) |
| **Görüntü Boyutu** | 640×640 | configs/train_detector_v5.yaml |
| **Split Politikası** | Group-aware (video_id, session, vehicle_id, person_id) | configs/data.yaml |
| **Veri Sızıntısı Kontrolü** | Geçti | reports/detector_v4_pretrain_leakage_check.json |

---

## 2. Train / Val / Test Split Detayları

### 2.1 Görüntü Sayıları

| Split | Görüntü | Etiket | Oran |
|-------|:-------:|:------:|:----:|
| **Train** | 10,998 | 10,998 | ~72.1% |
| **Val** | 2,855 | 2,855 | ~18.7% |
| **Test** | 1,395 | 1,395 | ~9.2% |
| **Toplam** | **15,248** | **15,248** | 100% |

> **Not:** Hedef split oranı `configs/data.yaml` içinde [0.70, 0.15, 0.15] olarak tanımlanmıştır. Gerçekleşen train oranı (72.1%) bu hedefi biraz aşmaktadır; bu durum group-aware splitting'in tam oran garantisi vermemesinden kaynaklanmaktadır.

### 2.2 Sınıf Bazlı Örnek Sayıları (Annotasyon / Instance)

| Sınıf ID | Sınıf Adı | Train | Val | Test | Toplam |
|:--------:|-----------|:-----:|:---:|:----:|:------:|
| 0 | telefonla_konusma | 1,907 | 500 | 250 | 2,657 |
| 1 | su_icme | 773 | 201 | 100 | 1,074 |
| 2 | arkaya_bakma | 785 | 202 | 101 | 1,088 |
| 3 | esneme | 762 | 197 | 102 | 1,061 |
| 4 | sigara_icme | 5,968 | 1,536 | 758 | 8,262 |
| 5 | emniyet_kemeri_ihlali | 1,260 | 186 | 93 | 1,539 |
| 6 | teknocan | 723 | 58 | 10 | 791 |
| 7 | bilgisayar | 662 | 50 | 6 | 718 |
| 8 | license_plate | 1,596 | 402 | 219 | 2,217 |
| **Toplam** | | **14,436** | **3,332** | **1,639** | **19,407** |

### 2.3 Sınıf Dengesizliği Analizi

```
sigara_icme (ID 4): 8,262 örnek — dominant sınıf (%42.6 tüm örneklerin)
teknocan    (ID 6):   791 örnek — 10.4× daha az (train'de 723)
bilgisayar  (ID 7):   718 örnek — 11.5× daha az (train'de 662)
```

**Dengeleme Stratejisi (detector_v5 öncesi vs sonrası):**

| Sınıf | v4 Öncesi Train | v5 Sonrası Train | Artış Oranı |
|-------|:---:|:---:|:---:|
| teknocan | ~90 | 723 | **+703%** |
| emniyet_kemeri | ~654 | 1,260 | **+93%** |
| bilgisayar | ~335 | 662 | **+98%** |

---

## 3. Sınıf Tanımları ve Etiket Tutarlılık Denetimi

### 3.1 Dört Kritik Kaynak Karşılaştırması

| Sınıf | data.yaml | FTR_CLASS_MAP | competition_contract.py | LIVE_CLASS_MAP |
|-------|:---:|:---:|:---:|:---:|
| telefonla_konusma | ✅ (FTR) | ✅ ID 0 | ✅ DRIVER_ACTIONS | ❌ (phone, İngilizce) |
| sigara_icme | ✅ (FTR) | ✅ ID 4 | ✅ DRIVER_ACTIONS | ❌ (cigarette) |
| teknocan | ✅ (FTR) | ✅ ID 6 | ✅ OBJECT_LABELS | ❌ (toy) |
| bilgisayar | ✅ (FTR) | ✅ ID 7 | ✅ OBJECT_LABELS | ❌ (laptop/computer) |
| license_plate | ✅ (FTR) | ✅ ID 8 | ❌ (ayrı alan) | ✅ ID 0 |

> **Önemli Not:** Canlı pipeline (LIVE_CLASS_MAP) İngilizce etiketler kullanır; FTR pipeline Türkçe kanonik etiketler kullanır. Bu ayrım bilinçli olup `src/class_registry.py` içinde belgelenmiştir. `LIVE_TO_FTR_LABEL` sözlüğü bu iki dünya arasında köprü kurar.

### 3.2 Potansiyel Tutarsızlık: Split Oranı

- `configs/data.yaml` tanımlar: `[0.70, 0.15, 0.15]`
- Gerçekleşen: `[0.721, 0.187, 0.092]`
- **Risk:** Test seti hedefin altında (%9.2 vs %15.0). Test metriklerin istatistiksel gücü azalır.

---

## 4. Veri Kaynakları

| Kaynak Adı | Yaklaşık Örnekler | Lisans | Durum |
|-----------|:---:|-------|-------|
| DriveAct dataset (sürücü davranış) | ~70k video klip | Akademik araştırma | Manifest SHA-256 kilitli |
| Cigarette Smokers (Kaggle) | ~8,000 görüntü | CC BY 4.0 | Corpus'a dahil |
| seatbelt_v1 | ~1,000 raw / 500 seçilmiş | CC BY 4.0 | Doğrulandı |
| Teknocan sentetik | Bloke | PNG FG gerektirir | Üretim beklemede |
| Bilgisayar / Laptop (çeşitli) | ~662 train | Karma | Corpus'a dahil |
| Manuel etiketleme | Değişken | Özel | Auditli |

---

## 5. Veri Artırma Yöntemleri

### 5.1 YOLOv8 Dahili Augmentasyon (train_detector_v5.yaml)

| Augmentasyon | Parametre | Değer |
|:---:|:---:|:---:|
| HSV ton değişimi | hsv_h | 0.015 |
| HSV doygunluk | hsv_s | 0.7 |
| HSV parlaklık | hsv_v | 0.4 |
| Döndürme | degrees | 10.0 |
| Ölçek | scale | 0.5 |
| Yatay çevirme | fliplr | 0.5 |
| Mozaik | mosaic | 1.0 (her epoch) |
| Mixup | mixup | 0.15 |
| Copy-Paste | copy_paste | 0.15 |

### 5.2 Albumentations Tabanlı Özel Augmentasyon

`scripts/augment_dataset.py` içinde Albumentations pipeline:
- Gaussian blur, motion blur
- Brightness/contrast/gamma değişimi
- JPEG sıkıştırma artefaktı simülasyonu
- Küçük nesne copy-paste (nadir sınıflar için)

---

## 6. Etiket Kalite Riskleri

| Risk | Açıklama | Etki | Mitigation |
|------|----------|------|-----------|
| **Yanlış etiket** | Manuel etiketleme hataları | Model yanlış davranış öğrenir | `scripts/audit_labels.py` ile kontrol edilmiş |
| **Sınır durumlar** | `sigara_icme` vs `su_icme` benzeri görsel karışıklık | Yanlış sınıflandırma | Sınıf bazlı güven eşikleri |
| **Sentetik veri bias** | Teknocan için sentetik görüntüler | Gerçek dünya genellemesi zayıflar | Bloke edildi; gerçek FG bekleniyor |
| **Az test desteği** | teknocan test=10, bilgisayar test=6 | Metrik güvenilirliği düşük | Daha fazla gerçek veri gerekli |

---

## 7. Kişi / Oturum Bazlı Split Kontrolü

`configs/data.yaml` içinde `group_keys` tanımı:
```yaml
group_keys: [video_id, capture_session, vehicle_id, person_id]
```

Bu ayar, aynı sürücünün/aracın görüntülerinin train ve test'e dağılmasını (veri sızıntısı) engeller. `reports/detector_v4_pretrain_leakage_check.json` içinde:
```json
{"leakage_detected": false}
```

---

## 8. Veri Yetersizliği Uyarıları

> [!WARNING]
> **teknocan (ID 6):** Test split'inde yalnızca **10 örnek** bulunmaktadır. Bu, istatistiksel güvenilir AP/Recall değerleri için yetersizdir. Raporda bu metrik için "N=10, istatistiksel güvenilirlik sınırlıdır" notu eklenmelidir.

> [!WARNING]
> **bilgisayar (ID 7):** Test split'inde yalnızca **6 örnek** bulunmaktadır. AP50=0.7517 değeri bu kadar az örnekle %95 güven aralığı çok geniştir.

> [!NOTE]
> **OCR Test Seti:** Mevcut OCR değerlendirmesi 14 sentetik görüntü üzerinde yapılmıştır (CER=0.0, exact_match=100%). Bu, gerçek koşulları temsil etmemektedir. Gerçek plaka veri seti ile doğrulama eksiktir.

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
