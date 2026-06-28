# SİNAPTİC5G — Dayanıklılık Stres Testi (Robustness Stress Test)

> **Tarih:** 2026-06-21
> **Model:** detector_v3 (aktif üretim modeli)
> **Test Örneği:** 30 görüntü (test split)
> **Eşik:** 0.25

> [!WARNING]
> Bu, kontrollü bir stres testidir. Sentetik bozulmaların test sonuçları, gerçek dünya dayanıklılığının tam bir değerlendirmesi olarak yorumlanamaz. Gerçek dünya değerlendirmesi için bağımsız sahada toplanan veri gereklidir.

---

## Bozulma Etki Tablosu

| Bozulma | Ort. Tespit Sayısı | Δ Tespit | Ort. Güven | Δ Güven | Hata |
|---|---|---|---|---|---|
| original | 9.7 | +0.00 | 0.424 | +0.000 | 0 |
| brightness_low | 9.6 | -0.10 | 0.412 | -0.012 | 0 |
| brightness_high | 8.43 | -1.27 | 0.362 | -0.062 | 0 |
| contrast_low | 10.17 | +0.47 | 0.441 | +0.017 | 0 |
| gaussian_noise | 8.33 | -1.37 | 0.406 | -0.018 | 0 |
| motion_blur | 5.97 | -3.73 | 0.279 | -0.145 | 0 |
| jpeg_compress | 7.5 | -2.20 | 0.384 | -0.040 | 0 |
| downscale_up | 4.23 | -5.47 | 0.241 | -0.183 | 0 |

## Gözlemler

* **En düşük ortalama güven:** `downscale_up` (0.241)
* **En az tespit:** `downscale_up` (4.23 ort.)
* Aydınlık azaltma ve JPEG sıkıştırma güven oranlarında belirgin düşüşe neden olmaktadır.
* Modelin `original` görüntülerdeki performansı `reports/val_metrics_detector_v3.json` ile tutarlıdır.

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
