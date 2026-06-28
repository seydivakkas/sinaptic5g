# SİNAPTİC5G — Final Raporu İçin Hazır Bölümler

> **Sürüm:** 1.0  
> **Tarih:** 2026-06-25  
> **Yazar:** Seydi Eryılmaz (@seydivakkas)  
> **Not:** Bu bölümler doğrudan yarışma/akademik final raporuna eklenebilecek şekilde yazılmıştır.

---

## BÖLÜM 1 — Proje Özeti

SİNAPTİC5G, TEKNOFEST yarışması kapsamında geliştirilen, 5. nesil (5G) kablosuz ağ teknolojisi ile derin öğrenme tabanlı bilgisayarlı görüyü entegre eden akıllı araç yol güvenliği analiz platformudur. Sistem, araç içi kameradan elde edilen video akışını gerçek zamanlı olarak işleyerek sürücü davranış ihlallerini (telefonla konuşma, sigara içme, emniyet kemeri ihlali, uyuklama ve benzeri), dikkat dağıtıcı nesneleri (teknocan, bilgisayar), araç kimliğini (plaka, renk, tip) ve tehlikeli manevraları (slalom) otomatik olarak tespit etmekte ve yapılandırılmış bir JSON çıktısına dönüştürmektedir. Platform iki bağımsız çalışma modunu desteklemektedir: internet bağlantısı gerektirmeyen, Docker container tabanlı çevrimdışı FTR (Final Test Run) hattı ve CAMARA QoD (Quality on Demand) API entegrasyonlu canlı 5G MEC (Multi-Access Edge Computing) modu.

---

## BÖLÜM 2 — Sistem Mimarisi

SİNAPTİC5G, birbirine bağlı sekiz işlevsel katmandan oluşmaktadır. Veri katmanı, ham video akışını OpenCV ile kare kare okuyarak adaptif zamanlı örnekleme ve düşük ışık iyileştirmesi (Zero-DCE) uygular. Yapay zekâ katmanı, nesne ve davranış tespiti için YOLOv8m tabanlı özel model, araç sınıflandırması için COCO eğitimli YOLOv8n ve plaka okuma için LPRNet+CRNN zinciri olmak üzere üç ayrı sinir ağı modelini paralel koşturur. Takip katmanı, BoTSORT algoritması ile çok nesne takibini ve kuşbakışı (BEV) perspektif dönüşümü üzerinde çalışan Kalman filtresi ile hız tahmini gerçekleştirir. OCR katmanı, güven ağırlıklı zamansal oylama ve Levenshtein mesafesi tabanlı gruplama ile Türk plakalarını tanır. Karar katmanı, yedi farklı sinyal kaynağından ağırlıklı risk skoru üretir. API katmanı, CAMARA QoD durum makinesi ile 5G ağ kaynaklarını fayda güdümlü olarak tahsis eder. Çıktı katmanı, JSON Schema Draft 2020-12 doğrulaması ve atomik dosya yazımı ile şemaya uyumlu `results.json` üretir. Test katmanı, 79 otomatik pytest testi ve offline FTR kabul hattı ile sistem bütünlüğünü sürekli doğrular.

### Sistem Pipeline'ı

```
Video (MP4) → Adaptif Örnekleme → Zero-DCE Parlaklık İyileştirme
→ Lens Düzeltme → YOLOv8n (Araç) + YOLOv8m (Davranış)
→ BoTSORT + Kalman (Takip) → LPRNet + CRNN (Plaka OCR)
→ CNN-LSTM (Temporal Davranış) → Risk Skoru → CAMARA QoD
→ JSON Schema Doğrulama → results.json (Atomik)
```

---

## BÖLÜM 3 — Yapay Zekâ Çözümü

Sistemin temel yapay zekâ bileşeni, YOLOv8m (You Only Look Once, sürüm 8, medium varyant) mimarisine dayanan `detector_v5` modelidir. Model, 9 sınıfı tek geçişte tespit etmekte; ağ girdisi 640×640 piksel normalize görüntü tensörü olup çıkışı 8.400 tahmin hücresi içeren (1, 13, 8400) boyutlu tensördür. Eğitim, 15.248 görüntülük özel veri kümesi üzerinde 60 epoch boyunca NVIDIA RTX 4070 Laptop GPU (8GB VRAM, CUDA 12.6) ile gerçekleştirilmiştir. AdamW optimizörü (lr₀=0.001, weight_decay=0.0005) ve FP16 Automatic Mixed Precision (AMP) kullanılmıştır. Transfer öğreniminde backbone'un ilk 10 katmanı dondurularak ImageNet ağırlıkları korunmuş; yalnızca üst katmanlar görev-spesifik veriyle ince ayar yapılmıştır. Eğitim sonunda ONNX formatına (opset 17, FP16, basitleştirilmiş) dışa aktarılan model, SHA-256 kriptografik hash kontrolü ile kilitlenmiş ve üretim ortamına taşınmıştır.

Plaka okuma için iki aşamalı bir süreç uygulanmaktadır: LPRNet (Lisans Plaka Tanıma Ağı) önce plaka bölgesini saptamakta, ardından CRNN (Convolutional Recurrent Neural Network) CTC (Connectionist Temporal Classification) greedy dekodlama ile karakter dizisini üretmektedir. Tespit geçmişini zaman boyutunda değerlendirmek amacıyla CNN-LSTM mimarisi kullanılmaktadır; bu model, son 16 kareden çıkarılan 7 boyutlu özellik vektörlerini (göz kırpma oranı, ağız açıklık oranı, normalize hız, kafa açısı, telefon/sigara bayrağı, hız aşımı) analiz ederek sürücünün davranış sınıfını tahmin eder.

---

## BÖLÜM 4 — Veri Seti Oluşturma ve Bölme Stratejisi

Eğitim veri kümesi, DriveAct (sürücü davranış veri tabanı), Kaggle sigara içen sürücü görüntüleri ve özel toplanmış emniyet kemeri/teknocan/bilgisayar görüntülerini içeren heterojen kaynaklardan derlenmiştir. Toplam 15.248 görüntü, %72.1 eğitim, %18.7 doğrulama ve %9.2 test olarak group-aware (grup-farkındalıklı) bölme stratejisiyle ayrılmıştır. Bu stratejide `video_id`, `capture_session`, `vehicle_id` ve `person_id` anahtarları kullanılarak aynı kaynağa ait görüntülerin farklı bölümlere sızması (veri sızıntısı) önlenmiştir. Düşük destekli sınıfların öğrenilmesi amacıyla Copy-Paste augmentasyon stratejisi uygulanmış; bu sayede teknocan sınıfı eğitim desteği 90'dan 723'e (%703 artış), bilgisayar sınıfı 335'ten 662'ye (%98 artış) yükseltilmiştir.

---

## BÖLÜM 5 — Model Eğitimi

detector_v5 modeli, YOLOv8m önceden eğitilmiş ağırlıkları temel alınarak 60 epoch boyunca eğitilmiştir. Eğitim hiperparametreleri: başlangıç öğrenme oranı lr₀=0.001 (kosinüs azalım ile son değere lrf=0.01 × lr₀), momentum=0.937, weight_decay=0.0005, warmup_epochs=3. Kayıp fonksiyonu bileşenleri: box_loss (ağırlık=7.5), cls_loss (ağırlık=0.5), dfl_loss (ağırlık=1.5). Sınıf ağırlık üssü `cls_pw=0.3` ile düşük destekli sınıflara örtük ağırlık artırımı sağlanmıştır. Mozaik (mosaic=1.0), mixup (mixup=0.15) ve copy-paste (copy_paste=0.15) augmentasyonları eğitim boyunca aktif tutulmuş; son 10 epoch'ta mozaik kapatılarak (close_mosaic=10) hassas son uyum sağlanmıştır. En iyi doğrulama mAP50 değeri epoch 60'ta 0.9404 olarak kayıt altına alınmıştır.

---

## BÖLÜM 6 — Performans Değerlendirme

Test bölütü üzerinde gerçekleştirilen kapsamlı değerlendirme sonucunda detector_v5 modeli şu genel metriklere ulaşmıştır: Precision=0.9289, Recall=0.9169, F1=0.9229, mAP@0.50=0.9334, mAP@0.50:0.95=0.6687. Bu sonuçlar, önceki üretim modeli detector_v3'ün mAP@0.50=0.9164 değerini anlamlı biçimde aşmaktadır. Sınıf bazında en yüksek AP50 değerleri `telefonla_konusma` (%99.14) ve `su_icme`/`arkaya_bakma` (%99.50) sınıflarında elde edilmiştir. Kritik güvenlik sınıfları `emniyet_kemeri_ihlali` için Recall=0.8280 ve AP50=0.8595 değerleri elde edilmiş; bu değerler kabul kapısı eşiği olan Recall≥0.30'u önemli ölçüde aşmaktadır. Plaka okuma bileşeni değerlendirme test setinde %100 tam eşleşme oranı (CER=0.0) kaydetmiştir; ancak bu değerlendirme 14 sentetik görüntü üzerinde gerçekleştirilmiş olup gerçek dünya koşullarını temsil etmeyebilir. Sistem, 79 otomatik pytest testini başarıyla geçmiş ve offline Docker FTR kabul testini tamamlamıştır.

---

## BÖLÜM 7 — Özgün Yöntemler

Bu çalışmanın özgün katkıları şu beş başlıkta özetlenebilir: (1) **Adaptif Kare Örnekleme:** Tespit durumu, GPU varlığı ve video pozisyonuna bağlı olarak dinamik stride hesaplayan, hesaplama bütçesini tehlike anına göre yeniden dağıtan bir örnekleme mekanizması geliştirilmiştir. (2) **Sınıf-Spesifik Güven Eşikleri:** Her sınıf için bağımsız olarak threshold kalibrasyon çalışmasına dayanan güven eşiği tablosu oluşturulmuş; güvenlik-kritik sınıflarda düşük eşikle recall artırılırken yüksek FP üreten sınıflarda eşik yükseltilmiştir. (3) **Confidence-Weighted Temporal OCR Voting:** Son N OCR tahminini confidence ağırlığı ve Levenshtein mesafesi tabanlı gruplama ile birleştiren gürültüye dayanıklı plaka tanıma yöntemi tasarlanmıştır. (4) **BEV-Kalman Takip Entegrasyonu:** Kuşbakışı perspektif dönüşümü ile gerçek dünya koordinatlarında çalışan adaptif Kalman filtresi tabanlı çok nesne takip sistemi geliştirilmiştir. (5) **Kriptografik Model Kilitleme:** SHA-256 hash doğrulaması ve açık kabul kapısı kriterleriyle desteklenen model versiyonlama ve güvenlik süreci uygulanmıştır.

---

## BÖLÜM 8 — Matematiksel ve Algoritmik Dayanaklar

Nesne tespiti için YOLOv8m mimarisi anchor-free bounding box tahminini uygulamaktadır. Her hücre için sınıf olasılıkları `softmax(logits)` ile normalize edilir ve en yüksek olasılıklı sınıf `argmax` ile seçilir. Örtüşen tespitler Non-Maximum Suppression ile elenir: güven eşiği θ_conf=0.35, IoU eşiği θ_iou=0.45. Takip maliyet matrisi üç bileşeni birleştirir: `cost = 0.5×IoU + 0.3×BEV_proximity + 0.2×ReID_similarity`. Optimal eşleştirme scipy'nin Hungarian algoritması (O(n³)) ile bulunur. Risk skoru ağırlıklı doğrusal toplama ile hesaplanır; teorik maksimum 133 puan 100'e normalize edilir. Plaka karakterleri CTC greedy dekodlama ile üretilir: her zaman adımında argmax seçimi yapılarak blank token ve ardışık tekrarlar kaldırılır.

---

## BÖLÜM 9 — Test ve Doğrulama

Sistem doğrulaması üç katmanda gerçekleştirilmiştir: (1) **Birim Test Katmanı:** 15 test modülünden oluşan 79 pytest testi tam geçiş kaydetmiştir; bu testler yarışma kontrat uyumluluğu, QoD orkestratör durum geçişleri, WebRTC sinyalizasyon, stres ve OCR doğruluğunu kapsamaktadır. (2) **Offline FTR Kabul Testi:** Docker container içinde gerçek video girdisi (smoke_test.mp4) üzerinde uçtan uca pipeline başarıyla çalıştırılmış; JSON Schema Draft 2020-12 doğrulaması geçilmiş ve çıktı atomik olarak üretilmiştir. (3) **Model Bütünlük Doğrulaması:** Tüm üretim model dosyaları SHA-256 hash kilidiyle güvence altına alınmış; `verify_model_lock()` fonksiyonu her başlatmada bütünlük kontrolü gerçekleştirmektedir.

---

## BÖLÜM 10 — Sınırlılıklar ve Gelecek Çalışmalar

Mevcut sistemin başlıca sınırlılıkları ve gerçekleştirilen iyileştirmeler şunlardır:
(a) CPU E2E gecikme performansı başlangıçta 9.84 saniye düzeyindeyken, yapılan optimizasyonlar (lazy loading, CPU warmup bypass ve ORT SessionOptions thread optimizasyonları) ile 2.24 saniyeye düşürülmüş ve 8.0 saniyelik yarışma sınırı başarıyla karşılanmıştır.

### Elde Edilen Performans Kazançları (Ortalama E2E Süreleri)
| Mod | Optimizasyon Öncesi | Optimizasyon Sonrası | Yarışma Hedefi | Durum |
|---|:---:|:---:|:---:|---|
| **GPU (CUDA Fallback)** | 9.68 sn | **1.52 sn** | 8.0 sn | **GEÇTİ** ✅ |
| **CPU (CPU EP)** | 9.84 sn | **2.24 sn** | 8.0 sn | **GEÇTİ** ✅ |

(b) `sigara_icme` sınıfında recall değeri 0.7614 ile diğer sınıfların gerisinde kalmaktadır; bu durum büyük ölçüde sınıf dengesizliğinden (8.262 örnek, tüm örneklerin %42.6'sı) kaynaklanmaktadır. (c) `teknocan` ve `bilgisayar` sınıfları için test split desteği (sırasıyla N=10 ve N=6) istatistiksel güvenilir sonuçlar için yetersizdir. (d) Plaka OCR değerlendirmesi gerçek koşulları yansıtmayan sentetik test setiyle yapılmıştır. (e) Canlı 5G QoD entegrasyonu, gerçek Turkcell CAMARA API erişimi olmaksızın doğrulanamamıştır.

Gelecek çalışma olarak: gerçek plaka görüntüleriyle OCR doğrulama, teknocan sınıfı için gerçek nesne görüntüsü toplanması, CNN-LSTM temporal modelin gerçek sürücü verisiyle eğitimi ve CAMARA QoD canlı entegrasyon testleri planlanmaktadır.

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
