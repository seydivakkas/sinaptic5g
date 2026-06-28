# SİNAPTİC5G — Ekip İçin Açık Sorular ve Eylem Maddeleri

> **Sürüm:** 1.0  
> **Tarih:** 2026-06-25  
> **Hazırlayan:** Teknik Analiz Ajanı (AI)  
> **Öncelik:** 🔴 Kritik | 🟠 Yüksek | 🟡 Orta | 🟢 Düşük

---

## 1. ✅ ÇÖZÜLDÜ — CPU Gecikme Uyumluluğu

**Durum:** `detector_v5` modeli ve FTR hattı üzerinde yapılan performans optimizasyonları (lazy loading, CPU warmup bypass ve ORT SessionOptions thread optimizasyonları) ile CPU E2E gecikme süresi **2.24 saniyeye** düşürülmüş, 8.0 saniyelik limit başarıyla geçilmiştir.

### Elde Edilen Performans Kazançları (Ortalama E2E Süreleri)
| Mod | Optimizasyon Öncesi | Optimizasyon Sonrası | Yarışma Hedefi | Durum |
|---|:---:|:---:|:---:|---|
| **GPU (CUDA Fallback)** | 9.68 sn | **1.52 sn** | 8.0 sn | **GEÇTİ** ✅ |
| **CPU (CPU EP)** | 9.84 sn | **2.24 sn** | 8.0 sn | **GEÇTİ** ✅ |

---

## 2. 🔴 KRİTİK — teknocan ve bilgisayar Test Seti Yetersizliği

**Sorun:**
- `teknocan` (ID 6): Test split'inde **N=10** örnek
- `bilgisayar` (ID 7): Test split'inde **N=6** örnek

Raporda bildirilen AP50 ve Recall değerleri istatistiksel olarak güvenilir değildir. %95 güven aralığı bu kadar az örnekle çok geniştir.

**Ekip Soruları:**
1. Gerçek teknocan (Togg oyuncağı) görüntüleri toplanabilir mi? Minimum 50-100 test görüntüsü önerilir.
2. Gerçek bilgisayar/laptop in-car görüntüleri eklenebilir mi?
3. Raporda bu metrikleri bildirirken "N=10, istatistiksel güvenilirlik sınırlıdır" notu eklenecek mi?

**Risk:** Raporda %90 teknocan recall iddiası 10 örnekten geliyor. Jüri bu durumu sorgulayabilir.

---

## 3. 🟠 YÜKSEK — OCR Test Seti Gerçeklik Sorunu

**Sorun:** `reports/ocr_accuracy_test_results.json` içinde 14 test görüntüsünün tamamı `plate_001.jpg`–`plate_014.jpg` formatında sentetik görüntüler; tüm GT değerleri `34TR1001`–`34TR1014` formatında. Bu, gerçek dünya OCR performansını yansıtmıyor.

**Ekip Soruları:**
1. Gerçek araç plakası görüntüleri (GT etiketli) toplanabilir mi? 100-300 örnek önerilir.
2. `scripts/train_crnn_tr_plate.py` dosyası mevcut — CRNN modeli gerçek Türk plaka verisiyle fine-tune yapıldı mı?
3. `crnn.onnx` içindeki mevcut model hangi veriyle eğitildi?

**Risk:** Final raporda "OCR CER=0.00" iddiası jürinin şüpheyle karşılamasına neden olabilir.

---

## 4. 🟠 YÜKSEK — Canlı 5G / CAMARA Entegrasyon Durumu

**Sorun:** `reports/qod_decision_simulation.md` yalnızca deterministik simülasyon sonucunu gösteriyor. Gerçek Turkcell CAMARA API çağrısı yapılmamış.

**Ekip Soruları:**
1. Turkcell CAMARA sandbox erişimi sağlanabilir mi?
2. `live_5g_integration_status.md` güncel mi? Son durum nedir?
3. Raporda "canlı 5G entegrasyonu simülatörle doğrulanmıştır" ifadesi mi, "gerçek operatör ortamında test edilmiştir" mi kullanılacak?

**Öneri:** Simülasyonun gerçeği temsil ettiğini açıkça belgeleyip raporda "simülatör tabanlı doğrulama" olarak sunmak daha doğru olur.

---

## 5. 🟠 YÜKSEK — Eğitim Ortamı Tekrarlanabilirlik Kanıtı

**Sorun:** Eğitim logu `reports/detector_v5_training_log.md` içinde "Eğitim başarıyla tamamlandı" yazmakta fakat detaylı epoch-by-epoch kayıp ve metrik kayıtları `detector_v5_training_progress.json` dosyasında mevcut.

**Ekip Soruları:**
1. Eğitim komutu/scripti tekrarlandığında aynı sonuçları üretebiliyor mu? (seed=42, deterministic=true ayarlandı)
2. `models/runs/experiments/detector_v5_60ep/` dizininin içeriği (Ultralytics çıktısı) mevcut mu? Buradaki `results.csv` ve confusion matrix görüntüleri rapora eklenebilir mi?
3. GPU ortamı başkası tarafından tekrarlanabilir mi? Ortam tanımı (requirements.txt + CUDA sürümü) yeterli mi?

---

## 6. 🟡 ORTA — Sınıf Etiket Tutarsızlıkları

**Tespit Edilen Durum:**
- `LIVE_CLASS_MAP` (canlı pipeline): İngilizce etiketler (phone, cigarette, toy, togg, driver_face, license_plate)
- `FTR_CLASS_MAP` (yarışma pipeline): Türkçe kanonik etiketler (telefonla_konusma, sigara_icme, teknocan, ...)
- Bu fark `src/class_registry.py` içinde bilinçli olarak belgelenmiş

**Ekip Soruları:**
1. Canlı Android modeli (TFLite) `LIVE_CLASS_MAP` ile mi eğitildi? Bu modelin güncellenmesi gerekiyor mu?
2. `data.yaml` içinde `count: 6` yazıyor (canlı pipeline için) ama FTR model 9 sınıf eğitim yapıyor. Bu çelişkiyi yeni bir ekip üyesi yanlış anlayabilir — bir README güncellemesi gerekiyor mu?

---

## 7. 🟡 ORTA — Android TFLite Modeli Durumu

**Tespit Edilen Durum:** `models/model.tflite` mevcut (~51.8MB). `README.md` içinde bu model için `android_vehicle_sentinel` olarak tanımlanıyor.

**Ekip Soruları:**
1. Bu TFLite model detector_v5 ile mi güncellendi, yoksa hâlâ eski modeli mi yansıtıyor?
2. `scripts/export_android_tflite.py` son çalıştırıldığında hangi modeli export etti?
3. `models/android_vehicle_sentinel.manifest.json` içindeki tensor boyutu ve SHA-256 güncel mi?

**Risk:** Android uygulaması eski modeli kullanıyorsa canlı ve FTR sonuçları farklılık gösterebilir.

---

## 8. 🟡 ORTA — mAP@0.50:0.95 Düşüşü Açıklaması

**Tespit:** detector_v3 mAP@0.50:0.95=0.6878 iken detector_v5 0.6687 elde etmiştir (%2.8 düşüş).

**Ekip Soruları:**
1. Bu düşüşün nedeni araştırıldı mı? (Localization hassasiyeti mi, belirli sınıflar mı?)
2. Raporda bu düşüşü nasıl açıklayacaksınız? (Önerilen açıklama: "mAP@0.50 birincil yarışma metriğidir ve %1.86 artış sağlanmıştır; mAP@0.50:0.95 ise eğitim setinin lokalizasyon varyasyonlarından etkilenmiş olabilir")
3. Test split içindeki `sigara_icme` sınıfının AP50=0.8577 vs AP50-95=0.5076 uçurumu — localization quality'i artırmak için ek veri gerekiyor mu?

---

## 9. 🟢 DÜŞÜK — Teknocan Sentetik Veri Durumu

**Tespit:** `reports/teknocan_synthetic_blocked.txt` ve `reports/teknocan_blocker_report.md` dosyaları mevcuttur. Sentetik teknocan üretimi PNG foreground eksikliği nedeniyle bloke.

**Ekip Soruları:**
1. Gerçek teknocan PNG foreground görüntüleri temin edilebilir mi?
2. `scripts/generate_teknocan_synthetic.py` bekleyen bir PNG klasörü var mı?
3. Bloke kaldırıldığında ne kadar sentetik veri üretilecek?

---

## 10. 🟢 DÜŞÜK — Kamera Kalibrasyon Parametreleri

**Tespit:** `CameraCalibrator.load()` önce `configs/camera_calibration.json` dosyasını arar; bulamazsa tahminsel geri dönüş kullanır.

**Ekip Soruları:**
1. Yarışmada kullanılacak kameranın gerçek kalibrasyon katsayıları ölçüldü mü?
2. `configs/camera_calibration.json` dosyası mevcut mu? Mevcut değilse BEV hız tahmini doğruluğu düşük kalır.
3. Yarışma kamerası sabit mi? Varsa kalibrasyon yapılması önerilir.

---

## Hızlı Özet Eylem Listesi

| # | Öncelik | Aksiyon | Sorumlu | Süre Tahmini |
|---|---------|---------|---------|--------------|
| 1 | 🔴 | GPU (T4) ortamında e2e gecikme benchmarkı | DevOps | 1 gün |
| 2 | 🔴 | teknocan ve bilgisayar için gerçek test görseli | Veri Ekibi | 1-2 hafta |
| 3 | 🟠 | Gerçek plaka veri seti toplayıp OCR değerlendirmesi | Veri Ekibi | 1 hafta |
| 4 | 🟠 | Raporda simülasyon vs gerçek test ayrımını netleştir | Yazım | 1 gün |
| 5 | 🟠 | Android TFLite modelinin güncelliğini doğrula | Android Ekibi | 2 saat |
| 6 | 🟡 | mAP@0.50:0.95 düşüşü için açıklama bölümü yaz | ML Ekibi | 2 saat |
| 7 | 🟡 | data.yaml count=6 vs FTR 9-sınıf tutarsızlığı belgele | Dokümantasyon | 1 saat |
| 8 | 🟢 | Kamera kalibrasyon ölçümü yap | Donanım Ekibi | Yarışma öncesi |

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
