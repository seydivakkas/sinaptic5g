# SİNAPTİC5G — Canlı İzleme ve Kontrol Paneli (Demo Dashboard Snapshot)

> **Tarih:** 2026-06-21  
> **Mod:** Canlı 5G İzleme (WebRTC + QoD)  
> **Sistem Durumu:** ÇALIŞIYOR (Aktif Alımlar Alınıyor)  

---

## 📺 1. Canlı Video Akışı (WebRTC Video Feed)

```
+-------------------------------------------------------------+
| [🔴 CANLI - 5G] Cihaz: cam-01 | Konum: Ankara Yolu km 12    |
|                                                             |
|          Sürücü Bölgesi (YOLO + FaceMesh Bindings)          |
|          +---------------------------------------+          |
|          | Sürücü: Aktif (mAP: 0.91)             |          |
|          | Emniyet Kemeri: OK [Yeşil]            |          |
|          | Baş Pozisyonu: Karşıya Bakıyor        |          |
|          +---------------------------------------+          |
|                                                             |
|          Araç Sınıfı: sedan [01A0000] - 88 km/sa            |
|                                                             |
+-------------------------------------------------------------+
| Gecikme (E2E): 120ms | Çözünürlük: 1280x720 | FPS: 30       |
+-------------------------------------------------------------+
```

---

## 📡 2. 5G Ağ Kalitesi ve QoD Durumu (Network & QoD Status)

| Parametre | Değer | Durum |
|---|---|---|
| **Aktif Profil** | `QOS_L_BANDWIDTH` | ✅ AKTİF |
| **CAMARA Oturum ID** | `mock-session-1` | ✅ DOĞRULANDI |
| **Ağ Gecikmesi (Ping)** | 12 ms | Mükemmel |
| **Kare Kaybı Oranı** | 0.05% | Kararlı |
| **Fayda Skoru** | 0.815 | Kapı Geçildi |
| **Kalan Süre** | 85 saniye | Uzatılabilir |

---

## 🚨 3. Son İhlaller ve Olaylar (Recent Infractions)

| Zaman damgası | İhlal Türü | Tespit Güveni | Sürücü Eylemi / Plaka | Bildirim |
|---|---|---|---|---|
| 23:54:12 | Plaka Okuma | 98.2% | `01A0000` gri sedan | Loglandı |
| 23:53:45 | Sigara İçme | 87.5% | Sürücü bölgesi tespiti | 🚨 UYARI GÖNDERİLDİ |
| 23:51:04 | Hız İhlali | 91.0% | 88 km/sa (Limit: 50 km/sa) | 🚨 CEZA RAPORLANDI |

---

## 📊 4. Sistem Telemetrisi (System Telemetry)

* **CPU Kullanımı:** 28% (Adaptive Stride devrede: Stride=0.35s)
* **Bellek Kullanımı:** 1.2 GB
* **Model Çıkarım Süresi (ONNX):** Ortalama 56.4 ms
* **Zero-DCE İyileştirme Süresi:** 11.2 ms
* **OCR İşleme Süresi:** 59.7 ms

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
