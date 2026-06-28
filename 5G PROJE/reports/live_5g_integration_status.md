# Live 5G / WebRTC Entegrasyon Durum Raporu

> **Durum:** `UYARI` (Canlı entegrasyon testleri gerçek Turkcell ağı gerektirir)  
> **Tarih:** 2026-06-21  

## 1. Mimari Özet

SİNAPTİC5G iki bağımsız çalışma moduna sahiptir:

| Mod | Açıklama | Ağ Bağımlılığı |
|-----|----------|----------------|
| **Offline FTR** | `/app/data/input/video.mp4` → çıkarım → `/app/data/output/results.json` | `YOK` (tam izolasyon) |
| **Live 5G / WebRTC** | Android CameraX → WebRTC → GPU Media Servisi → BFF → QoD / NV / Redis | `VAR` (Turkcell CAMARA API) |

**Tasarım kuralı:** Ham video verisi asla BFF veya Redis üzerinden geçmez. Yalnızca SDP zarfları (`api/webrtc_signaling.py`) ve analiz sonuçları aktarılır.

## 2. BFF Bileşen Durumu

| Bileşen | Dosya | Durum |
|---------|-------|-------|
| WebRTC Sinyal Posta Kutusu | `api/webrtc_signaling.py` | `TAMAMLANDI` — SDP offer/answer döngüsü, Redis TTL, çakışma koruması |
| QoD Orchestrator | `api/qod_orchestrator.py` | `TAMAMLANDI` — Fayda ağırlıklı state-machine, hataları graceful Best-Effort'a devreder |
| CAMARA QoD Adapter | `api/qod_client.py` | `TAMAMLANDI` — Session create/delete/extend/reconcile; gerçek Turkcell auth gerektirir |
| Number Verification | `api/number_verification.py` | `TAMAMLANDI` — CAMARA NV v0.3 uyumlu |
| OAuth Token Manager | `api/oauth_manager.py` | `TAMAMLANDI` — Client Credentials akışı, otomatik token yenileme |
| Telemetry Hub | `api/telemetry_hub.py` | `TAMAMLANDI` — Yayın/abone mesajlaşması |

## 3. Unit Test Kapsamı

| Test Dosyası | Kapsam | Durum |
|---|---|---|
| `tests/test_webrtc_signaling.py` | SDP doğrulama, offer/answer döngüsü | ✅ GEÇTI |
| `tests/test_signaling_api.py` | BFF sinyal uç noktaları | ✅ GEÇTI |
| `tests/test_qod_orchestrator.py` | QoD state-machine, Best-Effort geri dönüşü | ✅ GEÇTI |
| `tests/test_bff_media_boundary.py` | Ham video BFF/Redis'e geçmeme kuralı | ✅ GEÇTI |
| `tests/test_telemetry_hub.py` | Telemetri hub yayını | ✅ GEÇTI |

## 4. Açık Engeller

> [!WARNING]
> Aşağıdaki konular gerçek Turkcell ağı ve CAMARA API sandbox erişimi olmadan doğrulanamaz.

- **Turkcell CAMARA QoD Sandbox:** Gerçek `client_id`, `client_secret`, QoS profil adları ve kota hesabı gerektirir.
- **Android CameraX WebRTC:** Üretim entegrasyon testi için fiziksel 5G cihazı ve GPU media servisi çalıştırmalı makine gerektirir.
- **Number Verification:** Turkcell operatör ağında gerçek SIM kartla doğrulama yapılamadan onaylanamaz.

## 5. Tasarım Güvenceleri

* Ham video akışı BFF veya Redis'e ulaşmaz — `test_bff_media_boundary.py` bu kuralı uygular.
* QoD hatası veya zaman aşımı durumunda sistem Best-Effort moduna geçer; çıkarım durmaz.
* Redis yalnızca SDP zarflarını depolar (TTL = 60 saniye).

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
