# SİNAPTİC5G — OCR Karakter Doğruluğu Değerlendirme Raporu

> **Tarih:** 2026-06-21
> **Tür:** Kuru Çalıştırma (Dry-run / Hazırlık Kiti)

> [!WARNING]
> Gerçek plaka test görüntüleri bulunamadığından, test simüle edilmiş veri ile gerçekleştirilmiştir. Gerçek ölçüm için `data/raw/ocr_test_images` dizinine görüntüleri yükleyip `data/raw/ocr_gt_template.csv` şablon dosyasını doldurunuz.

---

## Simüle Edilen Performans Metrikleri

* **Toplam Örnek Sayısı:** 3
* **Karakter Hata Oranı (CER):** 4.17% (Hedef: < 5%)
* **Tam Eşleşme Doğruluğu (Exact Match EM):** 66.67% (Hedef: > 90%)

## Değerlendirme ve Notlar
1. Plaka okuyucu (PlateRecognizer) CTC greedy decoding tabanlı CRNN mimarisi kullanmaktadır.
2. Türkçe plaka format kuralları (`TURKISH_PLATE_PATTERN`) sayesinde geçersiz okumalar elenmekte veya düzeltilmektedir.

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
