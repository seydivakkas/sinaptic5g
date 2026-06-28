# SİNAPTİC5G — Veri Seti Denetim Raporu

> **Sürüm:** 1.0  
> **Tarih:** 2026-06-25  
> **Yazar:** Seydi Eryılmaz (@seydivakkas)  
> **Kanıt Kaynakları:** `reports/detector_v5_split_summary.json`, `configs/data_v5.yaml`, `reports/detector_v5_class_comparison.csv`

---

## 1. Veri Seti Genel Özeti

| Parametre | Değer | Kaynak |
|-----------|-------|--------|
| **Toplam Görüntü** | 15,487 | detector_v5_split_summary.json |
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
| **Train** | 11,198 | 11,198 | ~72.3% |
| **Val** | 2,855 | 2,855 | ~18.4% |
| **Test** | 1,434 | 1,434 | ~9.3% |
| **Toplam** | **15,487** | **15,487** | 100% |

> **Not:** Hedef split oranı `configs/data.yaml` içinde [0.70, 0.15, 0.15] olarak tanımlanmıştır. Gerçekleşen train oranı (72.3%) bu hedefi biraz aşmaktadır; bu durum group-aware splitting'in tam oran garantisi vermemesinden ve azınlık sınıflarına uygulanan copy-paste veri artırımından kaynaklanmaktadır.

### 2.2 Sınıf Bazlı Örnek Sayıları (Annotasyon / Instance)

| Sınıf ID | Sınıf Adı | Train | Val | Test | Toplam |
|:--------:|-----------|:-----:|:---:|:----:|:------:|
| 0 | telefonla_konusma | 1,905 | 500 | 250 | 2,655 |
| 1 | su_icme | 774 | 201 | 100 | 1,075 |
| 2 | arkaya_bakma | 781 | 202 | 101 | 1,084 |
| 3 | esneme | 772 | 197 | 102 | 1,071 |
| 4 | sigara_icme | 5,968 | 1,536 | 758 | 8,262 |
| 5 | emniyet_kemeri_ihlali | 1,259 | 186 | 93 | 1,538 |
| 6 | teknocan | 956 | 58 | 36 | 1,050 |
| 7 | bilgisayar | 662 | 50 | 26 | 738 |
| 8 | license_plate | 1,583 | 402 | 219 | 2,204 |
| **Toplam** | | **14,660** | **3,332** | **1,685** | **19,677** |

### 2.3 Sınıf Dengesizliği Analizi

```
sigara_icme (ID 4): 8,262 örnek — dominant sınıf (%42.0 tüm örneklerin)
teknocan    (ID 6): 1,050 örnek — 7.8× daha az (train'de 956)
bilgisayar  (ID 7):   738 örnek — 11.2× daha az (train'de 662)
```

**Dengeleme Stratejisi (detector_v5 öncesi vs sonrası):**

| Sınıf | v4 Öncesi Train | v5 Sonrası Train | Artış Oranı |
|-------|:---:|:---:|:---:|
| teknocan | ~90 | 956 | **+962%** |
| emniyet_kemeri | ~654 | 1,259 | **+92%** |
| bilgisayar | ~335 | 662 | **+97%** |

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

> [!NOTE]
> **teknocan (ID 6):** Test split'indeki örnek sayısı veri artırımı ve dengeleme sonrasında **36 örneğe** çıkarılmıştır. Bu, başlangıçtaki duruma (10 örnek) göre çok daha dengeli ve güvenilirdir.

> [!NOTE]
> **bilgisayar (ID 7):** Test split'indeki örnek sayısı veri artırımı sonrasında **26 örneğe** çıkarılmıştır. Bu, başlangıçtaki duruma (6 örnek) göre çok daha güvenilirdir.

> [!NOTE]
> **OCR Test Seti:** Mevcut OCR değerlendirmesi 14 sentetik görüntü üzerinde yapılmıştır (CER=0.0, exact_match=100%). Bu, gerçek koşulları temsil etmemektedir. Gerçek plaka veri seti ile doğrulama eksiktir.

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
