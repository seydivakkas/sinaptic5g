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

---

# 5G & YAPAY ZEKÂ İLE AKILLI YOL GÜVENLİĞİ YARIŞMASI

## FİNAL TASARIM RAPORU (SÜRÜM 5)

**Proje:** SİNAPTİC5G  
**Takım Adı:** SinapticLink5G  
**Takım ID:** 989764  
**Başvuru ID:** 5205583  
**Tarih:** 25 Haziran 2026  
**TEKNOFEST 2026 - Havacılık, Uzay ve Teknoloji Festivali**

---

## İÇİNDEKİLER

1. [Proje Özeti](#1-proje-ozeti-5-puan)  
2. [Veriseti Oluşturulması](#2-veriseti-olusturulmasi-20-puan)  
3. [Yapay Zekâ Çözümü](#3-yapay-zeka-cozumu-50-puan)  
   3.1. [Problemin Analizi](#31-problemin-analizi-15-puan)  
   3.2. [Çözüm Mimarisi](#32-cozum-mimarisi-15-puan)  
   3.3. [Çözüm Detayları](#33-cozum-detaylari-20-puan)  
4. [Çözümün Sınanması](#4-cozumun-sinanmasi-20-puan)  
5. [Kaynakça](#5-kaynakca-5-puan)

---

## 1. PROJE ÖZETİ (5 PUAN)

SİNAPTİC5G, yol kenarı veya araç kamerasından alınan videoda ana aracı, gövde rengini, Türkiye plakası bölgesini ve sürücünün riskli davranışlarını analiz eden; sonuçları yarışmanın zorunlu `results.json` sözleşmesine dönüştüren uçtan uca bir yol güvenliği platformudur. 

Sistem, iki bağımsız çalışma modunu desteklemektedir:
*   **Çevrimdışı FTR (Final Test Run) Kipi:** İnternet bağlantısı gerektirmeyen, Docker konteyner tabanlı çevrimdışı çalışma zamanı. `/app/data/input/video.mp4` girdisini işler ve sonucu atomik olarak `/app/data/output/results.json` yoluna yazar. Değerlendirme sürecinde dış servislere (Android, Redis, CAMARA, Turkcell SIM vb.) bağımlılığı yoktur.
*   **Canlı 5G / WebRTC Kipi:** Android CameraX istemcisinden alınan canlı video görüntüsünü WebRTC aracılığıyla takım GPU sunucusuna taşır. BFF katmanı ham video taşımaz; yalnız kimlik doğrulaması, SDP/ICE sinyalleşmesi ve QoD orkestrasyonunu yürütür.

| Çalışma Kipi | Açıklama | Amacı | Dış Bağımlılıklar |
|---|---|---|---|
| **Çevrimdışı FTR Kipi** | Docker ortamında yerel video dosyasını işler ve `results.json` üretir. | Hakem değerlendirme ortamında kesintisiz ve kararlı çıktı üretmek. | Yok (Tümüyle offline) |
| **Canlı 5G Kipi** | Android cihazdan alınan kamera görüntüsünü WebRTC ile GPU sunucusuna taşır, telemetriyi geri yansıtır. | Gerçek zamanlı yol güvenliği senaryosunda düşük gecikme ve 5G ağ kalitesi (QoD) yönetimi. | Redis, FastAPI BFF, CAMARA APIs |

### Kullanılan Yapay Zekâ Yaklaşımı

Sistemde iki ana dedektör modeli paralel çalıştırılır:
1.  **YOLOv8n (COCO):** Genel araç, kişi ve bilgisayar adaylarının tespiti için kullanılır.
2.  **Özel YOLOv8m (SÜRÜCÜ DAVRANIŞI & PLAKA):** 9 sınıfı tek geçişte tespit etmek üzere eğitilmiş özel modeldir. Sınıf listesi şöyledir:
    *   *Sürücü Eylemleri:* `telefonla_konusma`, `su_icme`, `arkaya_bakma`, `esneme`, `sigara_icme`, `emniyet_kemeri_ihlali`
    *   *Nesneler:* `teknocan`, `bilgisayar`
    *   *Plaka:* `license_plate`

### Projede 5G'nin Rolü

Canlı kipte 5G, düşük gecikmeli WebRTC video aktarımı, Number Verification (kullanıcı/cihaz doğrulaması) ve Quality on Demand (QoD - dinamik ağ kalitesi) API yönetimi için kullanılır. QoD kararı rastgele tetiklenmez; yalnızca hedef varlık mevcutsa, araç yaklaşma veya risk ihlali gösteriyorsa ve kalibre edilmiş fayda modeli olumlu sonuç veriyorsa ağ kalitesi yükseltilir.

### Projenin Güçlü Yanları

*   **İki Kipli Mimari:** Yarışma değerlendirme hattı dış servislere bağımlı değildir; canlı kullanım hattı ise 5G ve WebRTC ile gerçek zamanlı çalışmaya uygundur.
*   **Model Byte Kilidi:** [model_lock.json](file:///d:/SİNAPTİC/5G%20PROJE/model_lock.json) içindeki SHA-256 hash değerleri [ftr_main.py](file:///d:/SİNAPTİC/5G%20PROJE/ftr_main.py) başlatıldığında doğrulanır. Modifiye edilmiş modellerin çalıştırılması engellenir.
*   **Şema Doğrulama:** Çıktı JSON dosyası resmi şemaya göre [validate_results_schema.py](file:///d:/SİNAPTİC/5G%20PROJE/scripts/validate_results_schema.py) aracılığıyla kontrol edilir. Enum dışı etiketler veya geçersiz güven skorları dışarı sızmaz.
*   **Atomik Çıktı Yazımı:** results.json dosyası önce geçici dosyaya yazılır, ardından `fsync` ve `os.replace` ile güvenli biçimde hedef dosyanın yerini alır.
*   **Sentinel Yaklaşımı:** OCR karakter tanıma doğruluğu kanıtlanmadığında jüriyi yanıltacak sahte plaka üretmek yerine `01A0000` sentinel değeri ve `0.0` güven skoru döndürülür.
*   **Takip Sistemi:** IoU, Kuşbakışı Mesafe (BEV), Renk Histogramı ve Kalman filtresi birlikte kullanılarak nesne kimliği kararlı tutulur.

### Ölçülmüş Başlıca Sonuçlar

| Ölçüm | Sonuç | Durum |
|---|:---:|---|
| **Özel Dedektör mAP@0.50** | **0.9334** | ÖLÇÜLDÜ ([detector_v5_test_metrics.json](file:///d:/S%C4%B0NAPT%C4%B0C/5G%20PROJE/reports/detector_v5_test_metrics.json)) |
| **Özel Dedektör mAP@0.50:0.95** | **0.6687** | ÖLÇÜLDÜ ([detector_v5_test_metrics.json](file:///d:/S%C4%B0NAPT%C4%B0C/5G%20PROJE/reports/detector_v5_test_metrics.json)) |
| **Precision** | **0.9289** | ÖLÇÜLDÜ ([detector_v5_test_metrics.json](file:///d:/S%C4%B0NAPT%C4%B0C/5G%20PROJE/reports/detector_v5_test_metrics.json)) |
| **Recall** | **0.9169** | ÖLÇÜLDÜ ([detector_v5_test_metrics.json](file:///d:/S%C4%B0NAPT%C4%B0C/5G%20PROJE/reports/detector_v5_test_metrics.json)) |
| **F1-Score** | **0.9229** | ÖLÇÜLDÜ ([detector_v5_test_metrics.json](file:///d:/S%C4%B0NAPT%C4%B0C/5G%20PROJE/reports/detector_v5_test_metrics.json)) |
| **Yazılım Regresyon Testleri** | **79/79 geçti** | ÖLÇÜLDÜ ([final_ftr_acceptance_report.md](file:///d:/S%C4%B0NAPT%C4%B0C/5G%20PROJE/reports/final_ftr_acceptance_report.md)) |
| **E2E CPU Dizi Süresi (Smoke)** | **2.24 sn** (8.0 sn hedefinin altında) | ÖLÇÜLDÜ ([detector_v5_e2e_latency_benchmark.json](file:///d:/S%C4%B0NAPT%C4%B0C/5G%20PROJE/reports/detector_v5_e2e_latency_benchmark.json)) |
| **E2E GPU Dizi Süresi (Smoke)** | **1.52 sn** (8.0 sn hedefinin altında) | ÖLÇÜLDÜ ([detector_v5_e2e_latency_benchmark.json](file:///d:/S%C4%B0NAPT%C4%B0C/5G%20PROJE/reports/detector_v5_e2e_latency_benchmark.json)) |
| **OCR Doğruluğu (Exact Match)** | **%100** | ÖLÇÜLDÜ (14 sentetik plaka, gerçek dünya varyasyonu HEDEF) |
| **T4 GPU Kabul Testi** | - | HEDEF (Resmi test ortamı erişimi bekliyor) |
| **Gerçek CAMARA / Turkcell Çağrısı**| - | HEDEF (Operatör entegrasyonu bekliyor) |

---

## 2. VERİSETİ OLUŞTURULMASI (20 PUAN)

SİNAPTİC5G modelinin eğitimi ve doğrulanması için lisansları izlenebilir, tekrarlanabilir ve heterojen bir veri kümesi stratejisi uygulanmıştır.

### 2.1. Veri Kaynakları

Model eğitimi için açık kaynaklı veri setleri ve yerel smoke test videoları birleştirilmiştir. Kullanılan veri kaynakları:

| Veri Kaynağı | Lisans / Durum | Kullanım Amacı |
|---|---|---|
| **Driver Distraction Detection** (Roboflow v3) | CC BY 4.0 | telefonla_konusma, su_icme, arkaya_bakma |
| **Driver Drowsiness YOLO** (Roboflow v4) | CC BY 4.0 | esneme / yorgunluk tespiti |
| **Cigarette Smokers** (Roboflow v5) | CC BY 4.0 | sigara_icme davranışı tespiti |
| **Detect Seatbelt** (Roboflow v7) | CC BY 4.0 | emniyet_kemeri_ihlali tespiti |
| **Turkish License Plate Dataset** (Kaggle 2776891) | CC0-1.0 | Türkiye plaka bölgesi tespiti |

*Yerel Video Onayı:* Projede kullanılan yerel test videolarının kişi/çekim onay kayıtları tamamlanana kadar bu veriler eğitim kümesine dahil edilmemiştir. Sadece smoke test ve entegrasyon amacıyla kullanılmıştır.

### 2.2. Etiketleme ve Anotasyon Stratejisi

Tüm nesneler YOLO formatında normalized bbox (`class x_center y_center width height`) olarak etiketlenmiştir. Kaynak veri kümelerindeki uyumsuz sınıf adları, eğitim öncesinde ortak bir etiket sözlüğüne dönüştürülmüştür:

| Kanonik Sınıf | Kaynak Etiket Örnekleri | Açıklama |
|---|---|---|
| `telefonla_konusma` | phone, calling, texting | Telefonla konuşma veya telefonla mesajlaşma |
| `su_icme` | drinking, drink | Sürücünün su veya içecek içmesi |
| `arkaya_bakma` | reaching_behind, looking_back | Sürücünün arkaya bakması veya arkaya uzanması |
| `esneme` | yawn, yawning | Sürücünün esneme veya yorgunluk belirtileri |
| `sigara_icme` | cigarette, smoking, vaping | Sürücünün sigara, e-sigara kullanması |
| `emniyet_kemeri_ihlali`| no_seatbelt, unbelted | Emniyet kemerinin takılı olmaması ihlali |
| `teknocan` | teknocan | Yarışmaya özel maskot / nesne |
| `bilgisayar` | laptop, computer | Kabin içi bilgisayar/laptop kullanımı |
| `license_plate` | plate, licence_plate | Araç plaka bölgesi |

Sınıf dönüşüm ve adaptasyon mantığı [competition_contract.py](file:///d:/SİNAPTİC/5G%20PROJE/src/competition_contract.py) modülü içinde gerçekleştirilir.

### 2.3. Veri Temizleme ve Hazırlama

1.  **Görüntü Filtreleme:** Okunamayan, bozuk veya anotasyon dosyası olmayan görüntüler veri kümesinden çıkarılmıştır.
2.  **Sınıf Dengeleme:** Eğitim histogramında `sigara_icme` sınıfı 9,102 örnekle çok baskındı. Bu dengesizliği önlemek için plaka veri seti en fazla 1,200 görüntü ile sınırlandırılmış; `teknocan` ve `bilgisayar` gibi azınlık sınıfları için **Copy-Paste Augmentation** (nesne kırpma ve sentezleme) uygulanmıştır. Bu yöntemle `teknocan` sınıfı eğitim desteği 90'dan 723'e (%703 artış), `bilgisayar` sınıfı ise 335'ten 662'ye (%98 artış) çıkarılmıştır.
3.  **Split Sızıntı Önleme:** Eğitim, doğrulama ve test bölmeleri oluşturulurken `video_id`, `capture_session` ve `person_id` bilgileri kullanılmış; aynı kişinin veya aynı oturumun farklı splitlere sızması önlenmiştir.
4.  **İzlenebilirlik:** Tüm veri dosyaları yolları, split bilgileri ve SHA-256 özetleri [training_dataset_manifest.yaml](file:///d:/SİNAPTİC/5G%20PROJE/reports/detector_v4_dataset_manifest.yaml) içinde kayıt altına alınmıştır.

### 2.4. Eğitim, Doğrulama ve Test Dağılımı

`detector_v5` modeli için kilitlenen nihai veri seti dağılımı şu şekildedir:

| Split | Görüntü Sayısı | Oran | Anotasyon (Instance) Sayısı |
|---|---:|---:|---:|
| **Eğitim (Train)** | 11.649 | %81,17 | 14.533 |
| **Doğrulama (Val)** | 1.559 | %10,86 | 2.313 |
| **Test (Test)** | 1.144 | %7,97 | 1.744 |
| **Toplam** | **14.352** | **%100** | **18.590** |

### 2.5. Veri Artırma (Augmentation) Teknikleri

*   **HSV Renk Dönüşümü:** Farklı ışık sıcaklıkları ve renk koşullarına dayanıklılık sağlamak için.
*   **Parlaklık / Kontrast Değişimi:** Gündüz/gece geçişleri ve gölge etkilerini simüle etmek için.
*   **Yatay Çevirme:** Sürücünün farklı açılardaki davranış simetrisini yakalamak için.
*   **Hareket Bulanıklığı (Motion Blur):** Hızlı araç hareketi ve düşük enstantane etkilerine karşı dayanıklılık için.
*   **Mosaic Augmentation:** YOLO eğitiminde nesne bağlamlarını birleştirmek için ilk 50 epoch açık tutulmuş; son 10 epoch'ta `close_mosaic=10` parametresiyle kapatılarak gerçek nesne sınırlarına uyum sağlanmıştır.

---

## 3. YAPAY ZEKÂ ÇÖZÜMÜ (50 PUAN)

### 3.1. Problemin Analizi (15 Puan)

Süreç boyunca karşılaşılan temel kök problemler ve uygulanan mühendislik çözümleri:

#### 3.1.1. Işık Değişimleri
*   *Problem:* Gece far parlaması, tünel giriş-çıkışları ve gölgeler araç rengini ve kabin içi küçük nesneleri kararsızlaştırır.
*   *Çözüm:* Eğitimde HSV ve kontrast artırımı uygulanmıştır. Araç rengi çıkarılırken RGB yerine HSV renk uzayı kullanılmış, dış gövdenin orta bandındaki ROI HSV medyan değeri alınmıştır. Düşük ışıklı sahnelerde aydınlatmayı artırmak amacıyla Zero-DCE modeli entegre edilmiş; aydınlık karelerde gereksiz işlem yapılmaması için **parlaklık eşikli conditional bypass** eklenmiştir.

#### 3.1.2. Hareket Bulanıklığı
*   *Problem:* Hızlı araç geçişleri ve kamera titreşimi küçük nesnelerde (plaka, telefon, sigara) bulanıklığa neden olur.
*   *Çözüm:* Girdi çözünürlüğü 640x640 olarak sabitlenmiş ve eğitim setine hareket bulanıklığı augmentasyonu eklenmiştir. Tek karelik hataları önlemek adına kararlar temporal (zamansal) izleme boyunca birleştirilerek verilir.

#### 3.1.3. Oklüzyon ve Nesne Örtüşmesi
*   *Problem:* Kabin içinde el, direksiyon, telefon ve emniyet kemeri birbirini örtebilir; yanal geçişlerde takip kimliği kopabilir.
*   *Çözüm:* Takip sisteminde sadece piksel IoU'su kullanılmamıştır. IoU'ya ek olarak kuşbakışı görünüm (BEV) düzlem mesafesi, HSV renk histogramı ve Kalman filtresi entegre edilmiştir. Kayıp izler belirli bir tampon süresince yaşatılır.

#### 3.1.4. Küçük Nesne ve Plaka Metni
*   *Problem:* Plaka kutusu konumlandırma ile karakter okuma (OCR) farklı zorluk derecelerine sahiptir.
*   *Çözüm:* YOLOv8m mimarisi tercih edilerek küçük nesne tespit performansı artırılmıştır. Dağıtım paketinde Türkçe karakter OCR ağırlığı bulunmadığından ve internetten dinamik kütüphane indirilmesi yasak olduğundan; OCR okumasının yapılamadığı durumlarda sistem rastgele plaka uydurmaz, şemanın zorunlu sentinel değeri olan `01A0000` ve `0.0` güven skoru çıktısını üretir.

#### 3.1.5. Değişken FPS ve Hız Kestirimi
*   *Problem:* Sabit kare sayısına dayalı hız hesaplamak, kare düşmesi durumlarında fiziksel olarak hatalı hız üretir.
*   *Çözüm:* Kutunun alt orta noktası homografi ile BEV düzlemine taşınır. Hız hesaplaması, ardışık kare sayıları üzerinden değil, gerçek zaman damgaları arasındaki fark (`dt`) kullanılarak yapılır. Kalibrasyon hatası riski nedeniyle homografi parametreleri sahneye özel kalibre edilmelidir.

#### 3.1.6. OCR Belirsizliği
*   *Problem:* Plaka OCR karakterleri küçük ve gürültülüdür.
*   *Çözüm:* LPRNet (plaka bölgesi tespiti) ve CRNN CTC greedy dekodlama kullanılmaktadır. Plaka çıktıları regex kontrolünden geçirilir.

---

### 3.2. Çözüm Mimarisi (15 Puan)

Platform veri ve kontrol düzlemleri birbirinden kesin çizgilerle ayrılmış modüler bir yapıya sahiptir.

```
Çevrimdışı FTR Akışı:
Video (MP4) → Frame Okuma (OpenCV) → Adaptif Stride → YOLOv8n + YOLOv8m (ONNX)
  → Takip ve Olay Agregasyonu (Kalman + BEV) → Plaka OCR (LPRNet+CRNN) 
  → JSON Schema Doğrulama (jsonschema) → results.json (Atomik)

Canlı 5G Akışı:
Android CameraX → WebRTC RTP/RTCP → GPU Medya Servisi (aiortc)
  → ONNX Runtime Inference → Olay Telemetrisi (JSON) → Android Arayüzü
```

| Bileşen | Teknoloji / Sürüm | Rolü | Sınırı |
|---|---|---|---|
| **Android İstemci** | CameraX 1.4.1, TFLite 2.14.0 | Canlı video akışı, yerel araç sentinel'i | İstemci sırrı, ham plaka saklamaz |
| **Canlı BFF** | FastAPI 0.115.0, Redis 7 | Kimlik doğrulama, QoD API orkestrasyonu | Video veya frame verisi taşımaz |
| **GPU Medya Servisi** | aiortc 1.14.0, ONNX Runtime GPU | WebRTC peer bağlantısı, inference, telemetri | Çevrimdışı FTR çıktısı üretmez |
| **FTR Konteyneri** | CUDA 12.1, ONNX Runtime | Tümüyle çevrimdışı video analizi ve JSON çıktısı | Canlı ağ, Redis, Android gerektirmez |

---

### 3.3. Çözüm Detayları (20 Puan)

#### 3.3.1. Model Eğitim ve Dışa Aktarım
Üretim modeli olan `detector_v5` (YOLOv8m), 640x640 giriş boyutunda eğitilmiş ve ONNX formatına (opset 17, FP16 precision, simplified) dışa aktarılmıştır. Model boyutları ve hash değerleri:
*   `models/detector.onnx` (49.47 MB): `da1840434b6a13eae5205ff5ce7e41c60edc52d93a11c0d422fdaa86a966d5b9`
*   `models/coco.onnx` (12.95 MB): `edb36998c56fa76df9127b4bc6159c991c6e1c9d81d2fb1b83d87a408fede4`

#### 3.3.2. Takip ve Olay Agregasyonu
Tespit edilen nesneler Kalman filtresi ve Hungarian eşleştirme algoritması ile takip edilir. Mesafe maliyet matrisi formülü:
$$\text{Cost} = 0.5 \times \text{IoU} + 0.3 \times \text{BEV\_Proximity} + 0.2 \times \text{ReID\_Similarity}$$
Tek karelik gürültülü eylemlerin JSON çıktısını kirletmesini önlemek amacıyla, eylemler zamansal olarak 1 saniyelik pencerelerde gruplanır (agregasyon). Segment içindeki en yüksek 5 tahminin ortalaması alınarak nihai güven skoru belirlenir.

#### 3.3.3. Plaka Normalizasyonu ve OCR
Plaka OCR işlemi LPRNet ve CRNN CTC greedy dekoder ile yürütülür. Okunan karakterler büyük harfe dönüştürülür, boşluklar kaldırılır, Türkçe karakterler İngilizce ASCII karşılıklarına normalize edilir (Ş->S, Ğ->G vb.) ve Türkiye plaka regex standardına göre doğrulanır. Regex doğrulaması geçilemezse plaka sentinel değeri `01A0000` olarak atanır.

#### 3.3.4. Donanım ve Yazılım Altyapısı
*   *Inference Engine:* ONNX Runtime 1.20.1 (CUDAExecutionProvider öncelikli, CPUExecutionProvider fallback).
*   *Inference Hızlandırma:* CPU üzerinde çalışma sırasında startup gecikmesini optimize etmek adına CPU warmup bypass edilir; thread sayıları `SessionOptions` üzerinden kısıtlanır.
*   *Çıktı Güvenliği:* [validate_results_schema.py](file:///d:/SİNAPTİC/5G%20PROJE/scripts/validate_results_schema.py) scripti ile JSON Schema doğrulandıktan sonra atomik dosya yazımı gerçekleştirilir.

---

## 4. ÇÖZÜMÜN SINANMASI (20 PUAN)

### 4.1. Test Ortamı ve Protokolü

Sistemin kalitesi üç bağımsız katmanda sınanmıştır:
1.  **Model Test Katmanı:** Bağımsız 1,144 test görüntüsü üzerinde dedektör metrikleri ölçülmüştür.
2.  **Yazılım Test Katmanı:** `pytest` ile 15 farklı test modülünü içeren **79 test senaryosu** koşturulmuştur.
3.  **Uçtan Uca Smoke Testi:** 4 saniyelik, 120 karelik örnek video üzerinde offline Docker pipeline'ının çalışması test edilmiştir.

### 4.2. Global Model Metrikleri

`detector_v5` üretim modelinin, önceki sürüm `detector_v3` ile karşılaştırmalı test split metrikleri:

| Metrik | detector_v3 (Eski) | detector_v5 (Aktif Üretim) | Değişim |
|---|:---:|:---:|:---:|
| **mAP@0.50** | 0.9164 | **0.9334** | +1.86% |
| **mAP@0.50:0.95** | 0.6878 | **0.6687** | -2.78% |
| **Precision** | 0.9225 | **0.9289** | +0.69% |
| **Recall** | 0.8905 | **0.9169** | +2.96% |
| **F1-Score** | 0.9062 | **0.9229** | +1.84% |

### 4.3. Sınıf Bazlı Metrikler (Test Split)

| Sınıf ID | Sınıf Adı | Örnek Sayısı (N) | Precision | Recall | AP50 | AP50-95 | Yorum / Bulgular |
|:---:|---|---:|---:|---:|---:|---:|---|
| 0 | `telefonla_konusma` | 250 | 0.9887 | 0.9760 | **0.9914** | 0.6938 | Yüksek kararlılık |
| 1 | `su_icme` | 100 | 0.9884 | 1.0000 | **0.9950** | 0.8307 | Kusursuz recall |
| 2 | `arkaya_bakma` | 101 | 0.9869 | 1.0000 | **0.9950** | 0.8367 | Yüksek AP50 |
| 3 | `esneme` | 102 | 0.9528 | 0.9895 | **0.9923** | 0.6264 | Kararlı davranış |
| 4 | `sigara_icme` | 758 | 0.9130 | 0.7614 | **0.8577** | 0.5076 | Sınıf içi varyasyon yüksek |
| 5 | `emniyet_kemeri_ihlali`| 93 | 0.9209 | 0.8280 | **0.8595** | 0.5887 | Kabul kapısını (0.30) geçti |
| 6 | `teknocan` | 10 | 0.9066 | 0.9000 | **0.9783** | 0.6722 | Copy-paste ile desteklendi |
| 7 | `bilgisayar` | 6 | 0.7526 | 0.8333 | **0.7517** | 0.4719 | Düşük destek, geliştirilmeli |
| 8 | `license_plate` | 219 | 0.9505 | 0.9639 | **0.9795** | 0.7900 | Bounding box doğruluğudur |

*Metrik Yorumu:* `teknocan` ve `bilgisayar` sınıflarının test setindeki destek sayıları düşüktür. Bu durum jüriye açıkça belirtilmiş ve gelecek veri toplama hedefi olarak tanımlanmıştır.

### 4.4. Yazılım Testleri

Tüm API, orkestratör, şema, plaka normalizasyonu ve model bütünlük kontrollerini kapsayan **79/79 pytest testi** başarıyla geçmiştir (`PASSED`).

### 4.5. Smoke Test ve Performans Optimizasyonu

Yapılan optimizasyonlar (lazy loading, CPU warmup bypass ve SessionOptions thread optimizasyonu) öncesi ve sonrası uçtan uca ortalama video işleme süreleri:

| Çalışma Modu | Optimizasyon Öncesi | Optimizasyon Sonrası | Yarışma Hedefi | Durum |
|---|:---:|:---:|:---:|---|
| **GPU (CUDA Fallback)** | 9.68 sn | **1.52 sn** | 8.0 sn | **GEÇTİ** ✅ |
| **CPU (CPU EP)** | 9.84 sn | **2.24 sn** | 8.0 sn | **GEÇTİ** ✅ |

Optimized FTR pipeline, 8.0 saniyelik kritik sürenin oldukça altında çalışmaktadır.

### 4.6. OCR Doğruluk Testleri

| Metrik | Değer | Test Seti Tanımı |
|---|:---:|---|
| **Exact Match Rate** | 1.00 (%100) | 14 sentetik Türkiye plakası |
| **Mean CER (Karakter Hata Oranı)**| 0.00 | 14 sentetik Türkiye plakası |
| **CRNN Latency (p50)** | ~250.8 ms | CPU üzerinde tek plaka crop |

*Uyarı:* OCR testi sadece sentetik plaka görüntüleri üzerinde yapılmıştır; gerçek dünya koşullarındaki (kirli, eğik, gece) performansı doğrulanmamıştır ve HEDEF olarak işaretlenmiştir.

### 4.7. Çözümümüze Neden Güveniyoruz?

*   **Değişmez Bütünlük:** Tüm model dosyaları SHA-256 hash'leri ile kilitlidir.
*   **Hatasız Çıktı Garantisi:** JSON Schema doğrulaması ve atomik dosya yazımı sayesinde bozuk çıktı üretilmesi imkansızdır.
*   **Ağsız FTR Çalışması:** Değerlendirme konteyneri dış dünyaya kapalıdır, sunucu kesintilerinden etkilenmez.
*   **Yazılım Kalitesi:** 79 otomatik birim/regresyon testi ile pipeline koruma altındadır.
*   **Açıklanabilirlik:** Metriklerin kısıtları, düşük destekli sınıflar ve hedefler dürüstçe raporlanmıştır.

---

## 5. KAYNAKÇA (5 PUAN)

1.  Jocher, G., Chaurasia, A., Qiu, J. (2023). *Ultralytics YOLO*. GitHub Repository. URL: https://github.com/ultralytics/ultralytics. Erişim: 20.06.2026. (YOLOv8m model eğitimi için kullanılmıştır).
2.  Microsoft. (2026). *ONNX Runtime Documentation*. URL: https://onnxruntime.ai/docs/. Erişim: 20.06.2026. (Model çıkarımı için kullanılmıştır).
3.  OpenCV. (2026). *Open Source Computer Vision Library*. URL: https://docs.opencv.org/. Erişim: 20.06.2026. (Görüntü işleme ve video okuma için kullanılmıştır).
4.  CAMARA Project. (2026). *Quality on Demand API*. URL: https://github.com/camaraproject/QualityOnDemand. Erişim: 20.06.2026. (Canlı kip ağ yönetimi için kullanılmıştır).
5.  CAMARA Project. (2026). *Number Verification API*. URL: https://github.com/camaraproject/NumberVerification. Erişim: 20.06.2026. (Canlı kip numara doğrulaması için kullanılmıştır).
6.  Roboflow Universe. *Driver Distraction Detection*, v3, CC BY 4.0. URL: https://universe.roboflow.com. Erişim: 20.06.2026. (Telefon ve dikkat dağınıklığı verisi).
7.  Roboflow Universe. *Driver Drowsiness YOLO*, v4, CC BY 4.0. URL: https://universe.roboflow.com. Erişim: 20.06.2026. (Esneme verisi).
8.  Roboflow Universe. *Cigarette Smokers*, v5, CC BY 4.0. URL: https://universe.roboflow.com. Erişim: 20.06.2026. (Sigara içme verisi).
9.  Roboflow Universe. *Detect Seatbelt*, v7, CC BY 4.0. URL: https://universe.roboflow.com. Erişim: 20.06.2026. (Emniyet kemeri ihlal verisi).
10. Durcan, İ. *Turkish License Plate Dataset*, Kaggle ID 2776891, CC0-1.0. URL: https://www.kaggle.com. Erişim: 20.06.2026. (Türkiye plaka verisi).
11. TEKNOFEST / Turkcell. (2026). *5G ve Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması FTR Aşaması Teslim Dokümantasyonu*.

---

**Kanonik Kanıt Dizini:**  
*   Model Bütünlüğü: [model_lock.json](file:///d:/SİNAPTİC/5G%20PROJE/model_lock.json)  
*   Veri Seti: [detector_v4_dataset_manifest.yaml](file:///d:/SİNAPTİC/5G%20PROJE/reports/detector_v4_dataset_manifest.yaml)  
*   Model Metrikleri: [reports/](file:///d:/SİNAPTİC/5G%20PROJE/reports/)  
*   Test Sonuçları: [final_ftr_acceptance_report.md](file:///d:/S%C4%B0NAPT%C4%B0C/5G%20PROJE/reports/final_ftr_acceptance_report.md)

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
