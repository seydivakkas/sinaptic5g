# OCR Doğrulama ve Karakter Başarımı Engel Raporu (OCR Accuracy Blocker Report)

> **Durum:** `HEDEF`  
> **Tarih:** 2026-06-21  
> **Modül:** `plate_ocr.py` (LPRNet + CRNN + CTC)  

## Mevcut Durum Analizi

SİNAPTİC5G sisteminde plaka tespit bölgesi (bounding box) başarımı YOLO modeli kapsamında `license_plate` (sınıf 8) ile doğrulanmış ve plaka OCR ortalama gecikme süresi (latency) ölçülmüştür. Ancak, karakter seviyesinde bağımsız OCR doğruluğu (Exact Match) ve Karakter Hata Oranı (CER - Character Error Rate) ölçümleri için ayrılmış bir test seti bulunmamaktadır. Dolayısıyla, karakter tanıma başarımı resmi olarak **`HEDEF`** durumundadır.

### Sentinel / Voting Güvenlik Mekanizması

OCR çıktılarının kararlılığını artırmak ve anlık karakter okuma hatalarını sönümlemek adına sistemde tampon tabanlı bir **Sentinel / Oylama (Voting)** mekanizması çalışmaktadır:
* Okunan son plaka dizileri `_plate_buffer` (boyut = 5) adı verilen bir tamponda tutulur.
* Tampondaki elemanların çoğunluk oylaması (`_get_voted_plate`) üzerinden en az 3 kez eşleşen plaka nihai çıktı olarak atanır.
* Hatalı veya standart dışı karakter dizileri `TURKISH_PLATE_PATTERN` (örn. `34ABC123`) regex filtresinden geçemediğinde reddedilir ve oylamada elenir.

## Ekip İçin Aksiyon Listesi (Checklist)

OCR karakter doğruluğunu `ÖLÇÜLDÜ` durumuna getirebilmek için sonraki aşamalarda yapılması gerekenler:

- [ ] En az 300-500 adet gerçek araç içi / yol kamerası açılı plaka kırpması (plate crops) toplanmalıdır.
- [ ] Bu kırpmalar için karakter karakter tam plaka dizileri (ground-truth plate strings) manuel olarak etiketlenmelidir (örnek CSV formatında).
- [ ] Karakter Hata Oranı (CER), Kelime Hata Oranı (WER) ve tam eşleşme (Exact Match) oranlarını hesaplayan bağımsız bir test betiği yazılarak doğrulama yapılmalıdır.

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
