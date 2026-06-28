# SİNAPTİC5G — Eşik ve Güven Kalibrasyonu Çalışması

> **Tarih:** 2026-06-21
> **Model:** detector_v3 (aktif üretim)

> [!IMPORTANT]
> Bu çalışma, çıkarım zamanı eşiklerinin sınıf bazlı optimize edilip edilemeyeceğini araştırmaktadır. Önerilen eşikler yalnızca öneri niteliğindedir; üretime uygulanması için tam FTR kabul testleri tekrar çalıştırılmalıdır.

---

## Sınıf Bazlı Önerilen Eşikler

| Sınıf | Mevcut Eşik | Önerilen Eşik | En İyi F1 | Durum |
|-------|------------|--------------|----------|-------|
| telefonla_konusma | 0.25 | 0.3 | 0.280 | ÖLÇÜLDÜ |
| su_icme | 0.25 | 0.85 | 0.462 | ÖLÇÜLDÜ |
| arkaya_bakma | 0.25 | 0.8 | 0.467 | ÖLÇÜLDÜ |
| esneme | 0.25 | 0.55 | 0.444 | ÖLÇÜLDÜ |
| sigara_icme | 0.25 | 0.75 | 0.256 | ÖLÇÜLDÜ |
| emniyet_kemeri_ihlali | 0.25 | 0.9 | 0.500 | ÖLÇÜLDÜ |
| teknocan | 0.25 | 0.25 | 0.000 | UYARI (düşük destek) |
| bilgisayar | 0.25 | 0.25 | 0.000 | UYARI (düşük destek) |
| license_plate | 0.25 | 0.85 | 0.420 | ÖLÇÜLDÜ |

## Notlar

* `teknocan` ve `bilgisayar` sınıfları için düşük örnek sayısı nedeniyle eşik önerileri güvenilir değildir (UYARI).
* `emniyet_kemeri_ihlali` için düşük recall sorunu eşik düşürülerek kısmen iyileştirilebilir; ancak bu FP artışına neden olur.
* Tam eşik kalibrasyonu için detector_v4 tam eğitim sonrası daha fazla test örneği ile tekrarlanmalıdır.

Tam CSV: `reports/threshold_calibration_study.csv`

---
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
