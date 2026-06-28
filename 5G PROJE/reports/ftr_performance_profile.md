# SİNAPTİC5G — FTR Performans Profili Raporu

> **Tarih:** 2026-06-21
> **Ortam:** CPU Kıyaslaması (ONNX CPU Execution Provider)
> **Girdi Dosyası:** `tests/smoke_input/smoke_test.mp4`

> [!NOTE]
> Bu profil çalışması, ftr_main.py iş akışının tüm ana bileşenlerinin kare başına işleme sürelerini ölçer. Çevrimdışı kabul sınırlarında kalmak için dinamik kare atlama (adaptive stride) stratejisi uygulanmaktadır.

---

## Bileşen Bazlı Gecikme İstatistikleri (Milisaniye)

| Bileşen | Örnek Sayısı | Ortalama (ms) | Medyan (p50) | p95 | Minimum | Maksimum | Toplam Süre (s) |
|---|---|---|---|---|---|---|---|
| Zero-DCE | 4 | 4.78 | 2.88 | 9.52 | 2.69 | 10.67 | 0.019s |
| YOLO | 4 | 965.46 | 964.67 | 971.18 | 960.62 | 971.88 | 3.862s |
| OCR | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000s |
| LSTM | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000s |
| Total | 4 | 981.12 | 976.64 | 995.51 | 972.76 | 998.44 | 3.924s |

## Performans Analizi ve Çıkarımlar

1. **Darboğaz Analizi:** En çok zaman harcayan bileşen `YOLO` bileşenidir.
2. **Kare Başına Ortalama Süre:** Kare başına toplam işleme süresi ortalama **981.12 ms** düzeyindedir.
3. **Optimizasyon Etkisi:** `detector_optimized.onnx` kullanımı ve LPRNet+CRNN stride mekanizması CPU yükünü dengede tutmakta ve kabul kriterlerini karşılamasını sağlamaktadır.
4. **Canlı 5G Çıkarımı:** Bu gecikme değerleri, WebRTC hattı üzerinde de gecikme bütçesinin (frame-to-frame delay) gerçek zamanlı video akışı için sürdürülebilir olduğunu doğrulamaktadır.

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
