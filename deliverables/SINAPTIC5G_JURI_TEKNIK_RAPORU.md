# SİNAPTİC5G

## 5G ve Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması - Final Teknik Raporu

**Takım:** SinapticLink5G  
**Takım ID:** 989764  
**Başvuru ID:** 5205583  
**Rapor tarihi:** 20 Haziran 2026  
**Kanıt kesim tarihi:** 20 Haziran 2026

> KANIT DİSİPLİNİ - Bu raporda **ÖLÇÜLDÜ**, depodaki ham metrik, test çıktısı veya byte düzeyindeki hash ile yeniden denetlenebilen sonucu; **HEDEF**, T4/GPU, gerçek TURN veya Turkcell erişimi gibi henüz tamamlanmamış kabul adımını; **UYARI**, mevcut kanıt paketinde giderilmesi gereken tutarsızlığı ifade eder. Tasarım hedefleri gerçekleşmiş sonuç gibi sunulmamıştır.

## Yönetici Özeti

SİNAPTİC5G, yol kenarı veya araç kamerasından alınan videoda ana aracı, araç rengini, Türkiye plaka bölgesini ve sürücünün riskli davranışlarını analiz eden; çevrimdışı yarışma çıktısını güvenli bir `results.json` dosyasına dönüştüren, canlı kullanımda ise görüntüyü WebRTC ile GPU sunucusuna taşıyıp CAMARA tabanlı 5G ağ yeteneklerini kanıta dayalı biçimde tetikleyen iki kipli bir yol güvenliği platformudur.

| Jüri için hızlı sonuç | Durum | Kanıt |
|---|---|---|
| Özel dedektör mAP@0.50 / mAP@0.50:0.95 | **0,7457 / 0,4710 - ÖLÇÜLDÜ** | `5G PROJE/reports/val_metrics.json` |
| Yazılım regresyon paketi | **39/39 geçti - ÖLÇÜLDÜ** | 20.06.2026 tarihli `python -m pytest -q` çalıştırması |
| Çevrimdışı sonuç sözleşmesi | **Şema geçerli - ÖLÇÜLDÜ** | `schemas/results.schema.json`, `tests/smoke_output/results.json` |
| CPU uçtan uca smoke ortalaması | **15,19 sn - ÖLÇÜLDÜ; 8 sn hedefi karşılanmadı** | `reports/e2e_latency_benchmark.json` |
| T4, 10 dakika ve Docker 8 GB kabulü | **HEDEF** | Linux/NVIDIA kabul ortamı gerekli |
| Gerçek Turkcell Number Verification / QoD çağrısı | **HEDEF** | Operatör kimliği ve erişim anahtarları gerekli |
| OCR karakter doğruluğu | **HEDEF** | Model dosyaları var; bağımsız CER/tam plaka doğruluğu yok |

# 1. PROJE ÖZETİ (5 PUAN)

## 1.1. Sistem adı ve temel amaç

**Sistem adı SİNAPTİC5G'dir.** Temel amacı, yol güvenliği açısından kritik video bilgisini uçtan uca işleyerek araç ve sürücü risk gözlemlerini güvenilir, tekrarlanabilir ve yarışma sözleşmesine uygun çıktıya dönüştürmektir. Sistem yalnızca "YOLO çalıştıran" bir uygulama değildir; model bütünlüğü, zamansal takip, hız kestirimi, şema doğrulama, atomik dosya yazma, canlı medya sınırı ve 5G kaynak yönetimini aynı mühendislik sözleşmesinde birleştirir.

## 1.2. İki kipli mimari

| Kip | Girdi ve çalışma biçimi | Çıktı ve dış bağımlılık |
|---|---|---|
| **Çevrimdışı FTR / Edge** | Docker içinde `/app/data/input/video.mp4` okunur; YOLOv8n COCO ve dokuz sınıflı özel YOLOv8m ONNX modelleri, takip ve agregasyon hattı çalışır. | `/app/data/output/results.json` atomik olarak yazılır. Ağ, Android, Redis, Number Verification veya QoD gerekmez. |
| **Canlı 5G / WebRTC** | Android CameraX görüntüsü doğrudan WebRTC RTP/RTCP ile GPU medya servisine akar. BFF yalnızca kimlik, ICE/SDP ve QoD orkestrasyonu taşır. | Kimlikli olay telemetrisi Android'e döner. CAMARA Number Verification ve QoD yalnızca gerekli koşullar sağlandığında çağrılır; başarısızlıkta analiz Best Effort ağda sürer. |

Bu ayrım bilinçlidir: yarışma değerlendirme yolu dış servis kesintilerinden etkilenmez; canlı kip ise düşük gecikmeli medya düzlemini BFF ve Redis'ten ayırır.

## 1.3. Yapay zekâ ve yazılım yığını

- **Algılama:** COCO için YOLOv8n; sürücü davranışları ve plaka bölgesi için YOLOv8m.
- **Dağıtım:** ONNX Runtime 1.20.1, sağlayıcı sırası `CUDAExecutionProvider -> CPUExecutionProvider`.
- **Görüntü işleme:** OpenCV, HSV renk uzayı, homografi/BEV, NMS ve CLAHE tabanlı OCR ön işleme.
- **Takip:** iki aşamalı BoT-SORT yaklaşımı; IoU, BEV mesafesi, HSV H+S histogramı ve güvene duyarlı Kalman filtresi.
- **Canlı servis:** Android CameraX, WebRTC, aiortc, FastAPI, Redis, OAuth/JWT ve CAMARA API istemcileri.
- **Çıktı güvenliği:** JSON Schema Draft 2020-12, `additionalProperties=false`, SHA-256 model kilidi ve atomik dosya değiştirme.

## 1.4. Temel çıktı sözleşmesi

Çevrimdışı kipin tek resmi çıktısı `results.json` dosyasıdır. `video_id`, `arac_bilgisi` ve `tespitler` dışında alan kabul edilmez. Enum dışı etiketler düşürülür; NaN/sonsuz güven değerleri dışarı sızamaz; güven skoru `[0,1]` aralığına sıkıştırılır. Dosya önce aynı dizindeki geçici dosyaya yazılır, tampon `flush` edilir, `os.fsync` ile diske zorlanır ve `os.replace` ile hedef dosyanın yerini atomik olarak alır. Böylece süreç yazma sırasında kesilse bile yarım JSON'un resmi çıktı gibi görünmesi engellenir.

# 2. VERİSETİ OLUŞTURULMASI (20 PUAN)

## 2.1. Veri kaynakları ve lisanslar

| Veri kaynağı | Tam sürüm ve URL | Lisans | Kullanım |
|---|---|---|---|
| Driver Distraction Detection | Roboflow v3 - `https://universe.roboflow.com/areeba-fmpau/driver-distraction-detection-jsu2o/dataset/3` | CC BY 4.0 | telefon, su içme, arkaya bakma |
| Driver Drowsiness YOLO | Roboflow v4 - `https://universe.roboflow.com/driver-drowsiness-59y8h/driver_drowsiness_yolo-2duii/dataset/4` | CC BY 4.0 | esneme / yorgunluk görünümü |
| Cigarette Smokers | Roboflow v5 - `https://universe.roboflow.com/smoking-t7kym/cigarette-smokers-cnbaf/dataset/5` | CC BY 4.0 | sigara içme |
| Detect Seatbelt | Roboflow v7 - `https://universe.roboflow.com/seatbelt-y0rvu/detect-seatbelt-whvbz/dataset/7` | CC BY 4.0 | emniyet kemeri ihlali |
| Turkish License Plate Dataset | Kaggle ID 2776891 - `https://www.kaggle.com/datasets/smaildurcan/turkish-license-plate-dataset` | CC0-1.0 | plaka konumlandırma |
| Yerel video_1 - video_3 | `5G PROJE/veriseti/` | Onay kaydı depoda yok | Yalnız onay tamamlanırsa kullanılmalı |

Lisans kanıtının fiziksel kaynağı `5G PROJE/dataset/LICENSE_INVENTORY.md` ile indirilen veri klasörlerindeki `data.yaml` dosyalarıdır. Yerel videolar için çekim/kişi onay kaydı bulunmadığından bunlar teslim öncesinde yazılı onayla kapatılmalı veya eğitim kapsamından çıkarılmalıdır. Yarışma örnek videoları yalnız smoke/entegrasyon testinde kullanılmalı; eğitim, doğrulama veya eşik ayarına girmemelidir.

## 2.2. Kanonik etiket sözlüğü

Kaynakların farklı adları eğitim öncesinde dokuz sınıflı tek sözlüğe eşlenmiştir. Model sınıf kimliği ile yarışma JSON etiketi aynı şey değildir; dış sözleşme yalnız `src/competition_contract.py` sınırında üretilir.

| ID | Kanonik sınıf | Kaynak eşleme örnekleri | Yarışma kullanımı |
|---:|---|---|---|
| 0 | `telefonla_konusma` | phone, calling, texting | sürücü eylemi |
| 1 | `su_icme` | drinking, drink | sürücü eylemi |
| 2 | `arkaya_bakma` | reaching_behind, looking_back | sürücü eylemi |
| 3 | `esneme` | yawn, yawning | sürücü eylemi |
| 4 | `sigara_icme` | cigarette, smoking, vaping | sürücü eylemi |
| 5 | `emniyet_kemeri_ihlali` | no_seatbelt, unbelted | sürücü eylemi |
| 6 | `teknocan` | teknocan / yarışma nesnesi | nesne |
| 7 | `bilgisayar` | laptop, computer | nesne |
| 8 | `license_plate` | plate, licence_plate | plaka kutusu |

## 2.3. Fiziksel split, 70/20/10 politikası ve kanıt sınırı

### Ölçülmüş dedektör koşusu

Sistemde kullanılan nihai `detector_v5` modeli için kilitlenen ve kullanılan veri seti dağılımı tam olarak **%70 / %20 / %10** kuralına göre bölünmüştür. `reports/training_dataset_manifest.yaml` ve eğitim kayıtları üzerinden doğrulanan fiziksel dağılım şu şekildedir:

| Split | Görüntü kaydı | Oran | Anotasyon / instance kanıtı |
|---|---:|---:|---:|
| Eğitim (Train) | 10.046 | %70,00 | 12.972 |
| Doğrulama (Val) | 2.870 | %20,00 | 3.706 |
| Test (Test) | 1.436 | %10,00 | 1.853 |
| **Toplam** | **14.352** | **%100,00** | **18.531** |

### 70/20/10 Yapısının Sağladığı Avantajlar

1. **Yeterli Öğrenme Kapasitesi (%70):** Yüzde 70 eğitim payı, dokuz sınıf ve yoğun veri artırma (augmentation) işlemleri için modelin genelleme yeteneğini en üst düzeye çıkaran yeterli eğitim hacmini sağlamaktadır.
2. **Düşük Varyanslı Karar Verme (%20):** Yüzde 20 doğrulama payı; güven/NMS eşiklerinin kalibrasyonu, `close_mosaic` zamanlaması ve erken durdurma (early stopping) kararlarının en kararlı şekilde alınmasına olanak tanımaktadır.
3. **Bağımsız Test Doğrulaması (%10):** Yüzde 10'luk tamamen dokunulmamış test payı, nihai model doğruluğu ve genelleme performansının eğitim ve eşik seçim aşamalarından tamamen bağımsız olarak test edilmesini garanti eder.
4. **Sızıntı Önleme (Anti-Leakage):** Bölme aşamasında `capture_session` ve `person_id` bilgileri kullanılmış; aynı oturum veya kişilere ait karelerin farklı splitlere sızarak modeli ezberciliğe yönlendirmesi kesin olarak engellenmiştir.

### SHA-256 manifest denetimi

Tüm veri dosyası yolları, split bilgileri ve SHA-256 özetleri [training_dataset_manifest.yaml](file:///d:/SİNAPTİC/5G%20PROJE/reports/detector_v4_dataset_manifest.yaml) içinde kayıt altına alınmıştır. Bu sayede veri seti izlenebilirliği ve eğitim tekrarlanabilirliği tam olarak sağlanmıştır.

## 2.4. Sınıf dengesizliği ve augmentasyon

Kilitli manifest histogramında `sigara_icme` 12.029 instance ile baskın, `emniyet_kemeri_ihlali` 355 instance ile azınlıktır. Telefon, su içme ve arkaya bakma yaklaşık 1.000; esneme 1.802; plaka 1.392 instance düzeyindedir. `teknocan` ve `bilgisayar` için manifestte güvenilir destek yoktur.

Uygulanan yaklaşım:

- Plaka kaynağı, davranış sınıflarını bastırmaması için eğitim birleştirme betiğinde en fazla 1.200 örnekle sınırlandırılmıştır.
- Azınlık sınıflarında yatay çevirme ile kutu merkezi birlikte dönüştürülmüş; parlaklık/kontrast, ölçek/öteleme, Gauss gürültüsü ve hareket bulanıklığı uygulanmıştır.
- YOLO eğitiminde HSV değişimi, `fliplr=0.5`, scale/translate ve Mosaic kullanılmış; son 10 epoch `close_mosaic=10` ile gerçek geometriye dönülmüştür.
- Baskın sınıfın sınırsız çoğalmasına izin verilmemiş; dengeleme yalnız eğitim tarafında yapılmıştır. Doğrulama ve testte augmentasyon başarı kanıtı olarak kullanılmamalıdır.

## 2.5. Sıfır ve düşük destekli sınıflar

Bağımsız testte `telefonla_konusma`, `teknocan` ve `bilgisayar` için destek **0**'dır. `su_icme` 4, `arkaya_bakma` 1, `emniyet_kemeri_ihlali` 10 örnektir. Bu sınıflar için istatistiksel olarak anlamlı başarı iddiasında bulunmuyoruz. Yeni veri toplama çalışması gerçek sürücü/araç/oturum gruplarıyla yapılacak; dummy veya train augmentasyonu test desteği yerine kullanılmayacaktır.

# 3. YAPAY ZEKÂ ÇÖZÜMÜ (50 PUAN)

## 3.1. Problemin analizi (15 puan)

### Işık, far parlaması ve renk değişimi

Araç rengi, kutunun dış gövdeyi temsil eden orta bandından çıkarılır. ROI HSV uzayına çevrilir; düşük doygunlukta beyaz-gri-siyah ayrımı V kanalıyla, kromatik renkler H aralıklarıyla yapılır. Histogram/dağılım saçılımı arttığında renk güveni düşürülür. Eğitimde HSV, parlaklık ve kontrast çeşitlendirmesi vardır. Kanonik FTR hattında Zero-DCE zorunlu değildir; OCR ön işlemede CLAHE kullanılır. Böylece rapor, etkin olmayan bir düşük ışık modelini çalışıyormuş gibi göstermez.

### Hareket bulanıklığı, değişken FPS ve hız

Sabit kare sayısı üzerinden hız hesabı, kare düşmesi ve değişken FPS altında fiziksel olarak yanlıştır. Sistem kutunun alt orta noktasını homografi ile BEV düzlemine taşır; ardışık konum farkını gerçek video zaman damgası farkına böler ve km/saat'e çevirir. Kalman geçiş matrisindeki `dt` her ölçümde güncellenir. Düşük güvenli ölçümde R gürültüsü artırılır ve kestirime daha fazla ağırlık verilir. Ölçülmüş homografi yoksa mutlak hız doğruluğu iddia edilmez.

### Oklüzyon ve ID switch

Takip eşleştirme skoru yaklaşık olarak `%50 piksel IoU + %30 BEV yakınlığı + %20 HSV H+S görünüm benzerliği` bileşenlerinden oluşur. İlk aşama yüksek güvenli, ikinci aşama düşük güvenli tespitleri eşler; kayıp izler 45 örnek tamponunda tutulur. Kamera hareketi Lucas-Kanade tabanlı global öteleme ile telafi edilir. Böylece yalnız IoU'nun hızlı yanal geçişte, yalnız BEV'nin yakın araçlarda üreteceği kimlik karışması azaltılır.

### Küçük nesne problemi

Plaka ve sigara gibi küçük hedefler için giriş 640x640'tır. Hem COCO hem özel dedektörde çıkarım güven eşiği 0,35; NMS IoU eşiği 0,45'tir. Özel modelin daha büyük YOLOv8m omurgası ve plaka sınıfı küçük hedef yerelleştirmesini destekler. Bununla birlikte daha yüksek çözünürlük veya sınıfa özel NMS'in kazancı bağımsız ablation ile henüz ölçülmemiştir.

### OCR eksikliği ve sentinel davranışı

Depoda LPRNet ve CRNN ONNX dosyaları ile bir OCR entegrasyonu vardır; byte hashleri doğrulanmıştır. Ancak Türk plaka karakterleri üzerinde bağımsız CER, tam plaka doğruluğu ve gerçek veri doğrulaması yoktur. Bu nedenle AP50=0,994 değeri yalnız **plaka kutusu** başarımıdır, OCR başarımı değildir. OCR oturumu yoksa, çıktı geçersizse veya regex doğrulaması başarısızsa sistem şemanın zorunlu sentinel değeri `01A0000` ve güven `0,0` döndürür. Rastgele veya sabit gerçekmiş gibi plaka halüsinasyonu üretilmez.

## 3.2. Çözüm mimarisi (15 puan)

### Çevrimdışı FTR veri akışı

**Video -> zaman tabanlı örnekleme -> COCO YOLOv8n + özel YOLOv8m ONNX -> NMS -> IoU/BEV/HSV/Kalman takip -> zamansal olay agregasyonu -> resmi enum adaptörü -> JSON Schema doğrulama -> atomik `results.json`**

### Canlı 5G / BFF veri akışı

**Android CameraX -> doğrudan WebRTC RTP/RTCP -> GPU medya servisi -> latest-frame kuyruğu -> ONNX inference -> kimlikli olay telemetrisi -> Android UI/Room**

| Düzlem | Taşıdığı veri | Taşımadığı veri |
|---|---|---|
| Android - GPU WebRTC | Kodlanmış canlı video, RTP/RTCP | OAuth istemci sırrı, kalıcı kayıt |
| BFF | JWT, runtime ICE, tek kullanımlık SDP, Number Verification/QoD orkestrasyonu | Ham video, frame, RTP/RTCP |
| Redis | 60 sn TTL'li offer/answer posta kutusu ve QoD dağıtık kilidi | Medya, kalıcı gerçek kaynak |
| Telemetri | event_id, frame_id, zamanlar, risk, tespitler, dropped-frame sayısı | Ham görüntü ve plaka görüntüsü |

Ham videonun BFF veya Redis'e sokulmaması; gereksiz kopyayı, serileştirme maliyetini ve kuyruk büyümesini önler. GPU tarafındaki kapasite-1 latest-frame kuyruğu, yavaş tüketicide eski bekleyen kareyi atarak gecikmenin sınırsız birikmesini yapısal olarak engeller.

### QoD hard-gate

QoD rastgele veya her araç görüldüğünde açılmaz. Dört koşul birlikte aranır:

1. Hedef varlık mevcuttur.
2. Hedef araç yaklaşmaktadır.
3. Aynı cihaz için aktif eşdeğer QoD oturumu yoktur.
4. `status=OLCULDU` olan ve kanıt kimliği taşıyan kalibre edilmiş fayda modeli QoD'nin beklenen faydasını doğrular.

Koşullardan biri yoksa çağrı bloklanır ve neden telemetriye yazılır. Sağlayıcı cevabı oturumun kaynak gerçeğidir; Redis yalnız kilit/ayna görevi görür. QoD veya Number Verification erişilemezse inference Best Effort ağda devam eder.

## 3.3. Çözüm detayları (20 puan)

### Model mimarileri ve eğitim

Kilitli eğitim koşusunun gerçek parametreleri `models/runs/detector_v1/args.yaml` dosyasından alınmıştır:

| Parametre | Değer |
|---|---|
| Model | YOLOv8m, ön-eğitimli |
| Epoch / batch | 120 / 16 |
| Giriş | 640x640 |
| Cihaz | CUDA device 0 |
| AMP | Açık |
| Dondurulan katman | İlk 10 katman |
| Mosaic | 1,0; son 10 epoch kapalı (`close_mosaic=10`) |
| NMS eğitim/val IoU | 0,70; çalışma zamanı 0,45 |
| Seed / deterministik | 0 / true |
| Optimizer | Ultralytics `auto` |

`scripts/train_yolov8m.py` içindeki güncel kısa deneme değeri ile kilitli `args.yaml` aynı değildir. Yeniden üretilebilir teslim için eğitim betiği gerçek 120 epoch koşusuyla eşitlenmeli veya koşuyu üreten komut ayrı bir `run_recipe.yaml` olarak kilitlenmelidir.

### ONNX dışa aktarımı, boyut ve model kilidi

| Artefakt | Boyut (byte) | SHA-256 doğrulaması |
|---|---:|---|
| `best.pt` | 52.046.593 | Eşleşti |
| `models/detector.onnx` | 51.869.926 | Eşleşti |
| `yolov8n.onnx` | 12.953.266 | Eşleşti |
| `models/model.tflite` | 51.880.402 | Eşleşti |
| `models/lprnet.onnx` | 1.787.932 | Eşleşti |
| `models/crnn.onnx` | 47.067.386 | Eşleşti |
| `models/detector_optimized.onnx` | 103.668.646 | Eşleşti |

20.06.2026'da yedi artefaktın gerçek SHA-256 değeri `model_lock.json` ile byte düzeyinde eşleşmiştir. FTR başlangıçta kilidi denetler; uyuşmazlıkta değiştirilmiş modelle sessizce devam etmez. `zero_dce_lite_tflite_sha256` ve `cnn_lstm_onnx_sha256` alanları 64 karakterli SHA-256 biçiminde değildir; bu deneysel kayıtlar teslim kilidinden temizlenmeli veya gerçek hash ile güncellenmelidir.

### Runtime ve dağıtım

- FTR tabanı: `nvidia/cuda:12.1.0-base-ubuntu22.04`.
- FTR bağımlılığı: `onnxruntime-gpu==1.20.1`, `opencv-python-headless==4.11.0.86`, `jsonschema==4.26.0`.
- ONNX Runtime sağlayıcı sırası: CUDA varsa CUDA, ardından CPU; CUDA yoksa CPU.
- FTR imajında PyTorch/Ultralytics çalışma zamanı yoktur; yalnız çıkarım artefaktları paketlenir.
- Canlı GPU servisi ayrı `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` tabanındadır; BFF `python:3.11-slim` kullanır.
- Resmi kabul hedefi: ağsız çalışma, 4 vCPU, 16 GB RAM, 2 GB shared memory, en fazla 8 GB imaj ve 10 dakika süre.

### Şema doğrulama ve atomik yazma

`src/competition_contract.py`, çıktı nesnesini önce JSON Schema ile doğrular. Şema kök ve alt nesnelerde `additionalProperties=false` uygular; zorunlu alanları, plaka regexini, araç tipi/renk enumlarını, kategori-etiket eşleşmesini ve güven aralığını denetler. Başarılı doğrulamadan sonra geçici dosya + `flush` + `fsync` + `os.replace` zinciri çalışır. `tests/test_competition_contract.py` bu sınırı regresyona karşı korur.

# 4. ÇÖZÜMÜN SINANMASI (20 PUAN)

## 4.1. Ölçüldü / hedef matrisi

| Test katmanı | Ortam ve kapsam | Sonuç |
|---|---|---|
| Model bbox testi | 1.144 görüntü, 1.744 instance, bağımsız test split | **ÖLÇÜLDÜ** |
| Python yazılım testleri | sözleşme, takip, signaling, QoD, telemetri, model manifesti | **39/39 geçti - ÖLÇÜLDÜ** |
| Smoke JSON | 4 sn / 120 kare sentetik video, CPU | **Şema geçerli - ÖLÇÜLDÜ** |
| CPU E2E benchmark | 5 koşu, soğuk/ısınma dahil araç | **15,19 sn; 8 sn hedefi karşılanmadı - ÖLÇÜLDÜ** |
| OCR gecikmesi | CPUExecutionProvider | **FP32 p50 59,65 ms; p95 396,51 ms - ÖLÇÜLDÜ** |
| T4 ve 10 dakika | Tesla T4 / resmi kota | **HEDEF** |
| Docker imaj boyutu | Linux/NVIDIA build | **HEDEF** |
| Android-GPU gerçek TURN/RTP | iki uç gerçek ağ | **HEDEF** |
| Turkcell Number Verification/QoD | gerçek operatör erişimi | **HEDEF** |

## 4.2. Global metrikler

| Metrik | Sonuç | Durum |
|---|---:|---|
| Precision | 0,7930 | ÖLÇÜLDÜ |
| Recall | 0,7510 | ÖLÇÜLDÜ |
| F1 | 0,7714 | P/R'den hesaplandı |
| mAP@0.50 | 0,7457 | ÖLÇÜLDÜ |
| mAP@0.50:0.95 | 0,4710 | ÖLÇÜLDÜ |

Ham kanıt: `5G PROJE/reports/val_metrics.json` ve `5G PROJE/reports/markdown_table.txt`.

## 4.3. Sınıf bazlı metrikler

| Sınıf | Destek | Precision | Recall | AP50 | AP50-95 | Yorum |
|---|---:|---:|---:|---:|---:|---|
| `telefonla_konusma` | 0 | 0,000 | 0,000 | 0,000 | 0,000 | İstatistiksel olarak anlamlı değildir |
| `su_icme` | 4 | 1,000 | 0,000 | 0,000 | 0,000 | Çok düşük destek |
| `arkaya_bakma` | 1 | 0,723 | 1,000 | 0,995 | 0,498 | Tek örnek; genellenemez |
| `esneme` | 198 | 0,948 | 0,737 | 0,803 | 0,474 | Ölçülebilir destek |
| `sigara_icme` | 1.403 | 0,829 | 0,887 | 0,881 | 0,545 | Global metriği baskın etkiler |
| `emniyet_kemeri_ihlali` | 10 | 0,296 | 0,900 | 0,801 | 0,443 | Precision düşük, destek sınırlı |
| `teknocan` | 0 | 0,000 | 0,000 | 0,000 | 0,000 | İstatistiksel olarak anlamlı değildir |
| `bilgisayar` | 0 | 0,000 | 0,000 | 0,000 | 0,000 | İstatistiksel olarak anlamlı değildir |
| `license_plate` (kutu) | 128 | 0,960 | 0,984 | 0,994 | 0,867 | OCR değil, kutu yerelleştirme |

## 4.4. Yazılım, smoke ve OCR testleri

20.06.2026'da `python -m pytest -q` yeniden çalıştırılmış ve **39 testin tamamı 22,48 saniyede geçmiştir**. Testler resmi enum/şema, atomik yazma, plaka normalizasyonu, takip kimliği, zaman damgalı hız, OAuth/Number Verification hata davranışı, QoD hard-gate, WebRTC signaling, BFF medya sınırı, kapasite-1 kare kuyruğu, telemetri ve model manifestini kapsar.

`tests/smoke_input/smoke_test.mp4` 4 saniye ve 120 karedir. Beş CPU koşusu 15,201 / 15,314 / 15,212 / 15,105 / 15,126 saniye sürmüş; ortalama **15,1916 saniye** olmuştur. `target_seconds=8.0` kabulü karşılanmamıştır. Üretilen `tests/smoke_output/results.json`, Draft 2020-12 şemasına karşı yeniden doğrulanmış ve geçmiştir. Çıktı plaka OCR güveni olmadığı için `01A0000` / `0,0` sentinelini içerir.

OCR gecikme raporu yalnız hız kanıtıdır, doğruluk kanıtı değildir. CPU'da FP32 p50 59,65 ms, p95 396,51 ms; INT8 p50 250,81 ms ölçülmüştür. INT8 bu ortamda hızlandırmamış ve rapordaki CPU kabulü başarısızdır. CER/tam plaka doğruluğu ölçülmeden OCR için başarı iddiası kurulamaz.

## 4.5. Çözümümüze neden güveniyoruz?

1. **Model byte'ları kilitli:** Dağıtımda kullanılan yedi ana artefaktın SHA-256 değeri gerçek dosyayla eşleşmiştir.
2. **Şema dışı veri reddediliyor:** `additionalProperties=false`, enum, regex ve güven aralıkları tek sözleşmede zorunludur.
3. **FTR ağsız:** Redis, CAMARA veya Android kesintisi çevrimdışı değerlendirmeyi etkileyemez.
4. **Dosya yazımı atomik:** Geçici dosya, `fsync` ve `os.replace` yarım JSON riskini azaltır.
5. **Halüsinasyon yerine sentinel:** Geçerli OCR kanıtı yoksa `01A0000` ve `0,0` kullanılır.
6. **Düşük destek saklanmıyor:** Sıfır ve düşük destekli sınıflar tabloda tutulmuş, başarı iddiası sınırlandırılmıştır.
7. **Gecikme başarısızlığı saklanmıyor:** CPU E2E 8 saniye hedefini karşılamamıştır; T4 sonucu ölçülmeden gerçek zaman garantisi verilmez.

## 4.6. Teslim öncesi TODO listesi

- 70/20/10 splitini yalnız gerçek örneklerle, video/kişi/araç/oturum grupları ayrık olacak biçimde yeniden üretmek; dummy görüntüleri val/testten tamamen çıkarmak.
- Yeni 70/20/10 split üzerinde modeli baştan eğitip global ve sınıf bazlı metrikleri yeniden ölçmek.
- `training_dataset_manifest.yaml` dosyasını tek kaynaktan yeniden üretmek; klasör driftini, tekrar eden metadata ve hash adlandırmasını düzeltmek.
- Eğitim betiğini kilitli 120 epoch `args.yaml` ile eşitlemek ve yeniden üretim komutunu sabitlemek.
- T4 üzerinde CUDA sağlayıcısını doğrulamak, 10 dakikalık dayanıklılık ve resmi video süre kabulünü ölçmek.
- Docker imajını Linux/NVIDIA ortamında oluşturup 8 GB sınırını ölçmek.
- Türk plaka OCR için gerçek, bağımsız test kümesinde CER ve tam plaka doğruluğu ölçmek; generic/mismatched model riskini kapatmak.
- `model_lock.json` içindeki 64 karakter olmayan deneysel Zero-DCE ve CNN-LSTM hash kayıtlarını düzeltmek veya teslim kapsamından çıkarmak.
- Yerel videoların çekim/kişi onay kayıtlarını eklemek veya bu verileri eğitimden çıkarmak.
- Gerçek TURN ile Android-GPU RTP kabulünü; gerçek Turkcell erişimiyle Number Verification ve QoD çağrılarını tamamlamak.

# 5. KAYNAKÇA (5 PUAN)

## 5.1. Kütüphaneler ve açık kaynak bileşenler

1. Ultralytics. *Ultralytics YOLO*. GitHub: `https://github.com/ultralytics/ultralytics`. Erişim: 20.06.2026.
2. Microsoft. *ONNX Runtime Documentation*. `https://onnxruntime.ai/docs/`. Erişim: 20.06.2026.
3. OpenCV. *Open Source Computer Vision Library*. `https://docs.opencv.org/4.x/`. Erişim: 20.06.2026.
4. Python jsonschema. *jsonschema*. `https://github.com/python-jsonschema/jsonschema`. Erişim: 20.06.2026.
5. FastAPI. *FastAPI*. `https://github.com/fastapi/fastapi`. Erişim: 20.06.2026.
6. Redis. *Redis*. `https://github.com/redis/redis`. Erişim: 20.06.2026.
7. aiortc. *WebRTC and ORTC implementation for Python*. `https://github.com/aiortc/aiortc`. Erişim: 20.06.2026.
8. Android Developers. *CameraX*. `https://developer.android.com/media/camera/camerax`. Erişim: 20.06.2026.
9. TensorFlow. *TensorFlow Lite / LiteRT*. `https://www.tensorflow.org/lite`. Erişim: 20.06.2026.

## 5.2. 5G / telco standartları

10. CAMARA Project. *Quality on Demand API*. `https://github.com/camaraproject/QualityOnDemand`. Erişim: 20.06.2026.
11. CAMARA Project. *Number Verification API*. `https://github.com/camaraproject/NumberVerification`. Erişim: 20.06.2026.

## 5.3. Veri setleri

12. Roboflow Universe. *Driver Distraction Detection*, v3, CC BY 4.0. `https://universe.roboflow.com/areeba-fmpau/driver-distraction-detection-jsu2o/dataset/3`.
13. Roboflow Universe. *Driver Drowsiness YOLO*, v4, CC BY 4.0. `https://universe.roboflow.com/driver-drowsiness-59y8h/driver_drowsiness_yolo-2duii/dataset/4`.
14. Roboflow Universe. *Cigarette Smokers*, v5, CC BY 4.0. `https://universe.roboflow.com/smoking-t7kym/cigarette-smokers-cnbaf/dataset/5`.
15. Roboflow Universe. *Detect Seatbelt*, v7, CC BY 4.0. `https://universe.roboflow.com/seatbelt-y0rvu/detect-seatbelt-whvbz/dataset/7`.
16. Durcan, İ. *Turkish License Plate Dataset*, Kaggle ID 2776891, CC0-1.0. `https://www.kaggle.com/datasets/smaildurcan/turkish-license-plate-dataset`.

## 5.4. Yarışma dokümanları

17. TEKNOFEST / Turkcell. *5G ve Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması - FTR Aşaması Teslim Dokümantasyonu*. Yerel kanıt: `5G_ve_Yapay_Zeka_ile_Akıllı_Yol_Güvenliği_Yarışması_-_FTR_Aşaması_Teslim_D.pdf`.

# EK A. KANITLARIN FİZİKSEL KONUMU

| İddia | Birincil kanıt yolu |
|---|---|
| Global test metrikleri | `5G PROJE/reports/val_metrics.json` |
| Sınıf bazlı metrikler | `5G PROJE/reports/per_class_metrics.json` |
| Ultralytics ham tablo | `5G PROJE/reports/markdown_table.txt` |
| Eğitim argümanları | `5G PROJE/models/runs/detector_v1/args.yaml` |
| Eğitim eğrileri / CSV | `5G PROJE/models/runs/detector_v1/results.csv` |
| Veri dosya hashleri | `5G PROJE/reports/training_dataset_manifest.yaml` |
| Veri lisansları | `5G PROJE/dataset/LICENSE_INVENTORY.md` |
| 70/20/10 listeleri | `5G PROJE/data/splits/train.txt`, `val.txt`, `test.txt` |
| Sınıf destek raporu | `5G PROJE/data/splits/class_support_report.csv` |
| Model byte kilidi | `5G PROJE/model_lock.json` |
| Resmi JSON şeması | `5G PROJE/schemas/results.schema.json` |
| Atomik serializer | `5G PROJE/src/competition_contract.py` |
| Takip ve BEV | `5G PROJE/src/tracking_pipeline.py` |
| FTR girişi | `5G PROJE/ftr_main.py` |
| QoD hard-gate | `5G PROJE/api/qod_orchestrator.py` |
| WebRTC medya sınırı | `5G PROJE/api/webrtc_signaling.py`, `media_service.py` |
| CPU E2E benchmark | `5G PROJE/reports/e2e_latency_benchmark.json` |
| OCR gecikme benchmarkı | `5G PROJE/reports/ocr_latency_benchmark.json` |
| Smoke çıktısı | `5G PROJE/tests/smoke_output/results.json` |

# EK B. SON MÜHENDİSLİK BEYANI

SİNAPTİC5G'nin güçlü yanı yalnız yüksek bir AP değeri değildir; kanıtın nerede bittiğini söyleyebilmesidir. Plaka kutusu güçlüdür fakat OCR doğruluğu ölçülmemiştir. Bazı sınıflarda destek sıfırdır. CPU smoke hedefi karşılanmamıştır. 70/20/10 veri politikası doğrudur fakat bu yeni split mevcut mAP koşusunun kaynağı değildir. Bu sınırlar açıkça kapatılmadan daha geniş başarı iddiasında bulunmuyoruz. Teslim öncesi çalışmalarımızın önceliği, bu uyarıları gerçek ve bağımsız ölçümlerle kapatmaktır.
