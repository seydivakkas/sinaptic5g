# SİNAPTİC5G — Hata Analizi Raporu (Error Analysis Report)

> **Tarih:** 2026-06-21
> **Model:** detector_v3 (aktif üretim modeli)
> **Not:** Bu analiz yalnızca bağımsız test verisi (`data/curated/detector_v3/test/`) kullanılarak yapılmıştır. Eğitim verisi dahil edilmemiştir.

---

## 1. İnceleme Özeti

| Metrik | Değer |
|--------|-------|
| İncelenen görüntü | 150 |
| Yanlış Pozitif (FP) | 43 |
| Yanlış Negatif (FN) | 59 |
| Düşük güven TP (&lt;0.65) | 42 |
| Yüksek güven TP (≥0.65) | 86 |

## 2. Sınıf Bazlı Hata Analizi

### Yanlış Pozitifler (Sınıfa Göre)

| Sınıf | FP Sayısı |
|-------|----------|
| sigara_icme | 26 |
| esneme | 8 |
| su_icme | 3 |
| emniyet_kemeri_ihlali | 2 |
| telefonla_konusma | 2 |
| arkaya_bakma | 1 |
| license_plate | 1 |

### Yanlış Negatifler (Sınıfa Göre)

| Sınıf | FN Sayısı |
|-------|----------|
| sigara_icme | 27 |
| telefonla_konusma | 23 |
| esneme | 4 |
| emniyet_kemeri_ihlali | 2 |
| bilgisayar | 2 |
| teknocan | 1 |

## 3. Başlıca Hata Modları

1. **Düşük destek sınıflarında güvenilmez tahmin** — teknocan (8 test örneği) ve bilgisayar (6 test örneği) sınıflarında istatistiki anlamlı değerlendirme yapılamamaktadır.
2. **Emniyet kemeri ihlali (sınıf 5) geri çağırım sorunu** — AP@0.50=0.774 iken recall=0.698 olarak ölçülmüştür. Bu sınıftaki 43 test örneği düşük destek sayısına işaret eder.
3. **Sigara içme (sınıf 4) sınıf içi varyans** — 758 test örneğiyle en büyük grubu oluşturur; FP bu sınıfta daha yüksek görülür.
4. **Küçük nesne gözden kaçırma** — Küçük veya uzak pozisyonlardaki nesneler için FN oranı daha yüksektir.
5. **Sınıflar arası karışıklık** — telefonla_konusma / bilgisayar sınıfları görsel örtüşme nedeniyle karıştırılabilir.

## 4. Öneriler

* Sınıf 5 (emniyet kemeri ihlali) için detector_v4 tam eğitim tamamlandığında recall artışı beklenmektedir.
* Teknocan (sınıf 6) ve bilgisayar (sınıf 7) için veri artışı önceliklendirilmelidir.
* Büyütme augmentasyonlarına küçük nesne senaryoları eklenebilir.

## 5. Galeri

Görsel örnekler `reports/error_gallery/` altında düzenlenmiştir:
* `false_positive/` — FP örnekleri
* `false_negative/` — FN örnekleri
* `low_confidence_correct/` — Düşük güven TP örnekleri
* `high_confidence_correct/` — Yüksek güven TP örnekleri

Tam galeri indeksi: `reports/error_gallery/gallery_index.json`

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
