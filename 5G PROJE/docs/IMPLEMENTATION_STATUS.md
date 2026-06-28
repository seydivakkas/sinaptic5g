# Güncellenmiş Implementasyon Durumu — 20 Haziran 2026

| İstek | Kod karşılığı | Durum |
|---|---|---|
| Enum/`AnalysisResult` korunması | Mevcut sınıflar korunur; manifest yalnız `vehicle` eşler | Tamamlandı |
| Modelden bağımsız Android parser | İki tensor yönü, dinamik `C`, manifest labels/mapping | Tamamlandı |
| Tekrarlanabilir FP16 TFLite | Ultralytics 8.4.41, kaynak/çıktı hash’i, sabit komut | Tamamlandı |
| Build model kabulü | SHA-256, boyut, tensor lock, 1 TFLite/0 ONNX | Tamamlandı |
| BFF SDP posta kutusu | Auth, sahiplik, TTL, tek kullanım, 64 KiB, Redis 503 | Tamamlandı |
| BFF medya sınırı | WebSocket/frame/media ingest rotası yok | Tamamlandı |
| GPU WebRTC peer | `aiortc==1.14.0`, offer poll, answer, doğrudan video track | Tamamlandı |
| Latest-frame queue | Peer başına kapasite 1 ve dropped counter | Tamamlandı |
| GPU telemetri | JWT + cihaz sahipliği, `event_id` snapshot | Tamamlandı |
| Android WebRTC | Paket pinli; CameraX YUV aynı oturumdan video track’e beslenir | Tamamlandı |
| Room idempotency/veri minimizasyonu | `eventId` PK, migration/upsert; plaka/SDP/token/görüntü kolonu yok | Tamamlandı |
| Gerçek TURN/iki uç RTP kabulü | Dış ICE/TURN ve deploy ortamı bekleniyor | HEDEF |
| Özel dedektör/KPI | Bağımsız test: mAP50 0,7457; mAP50-95 0,4710; F1 0,7714 | ÖLÇÜLDÜ |
| Türk plaka OCR, sıfır destekli sınıflar ve T4 gecikmesi | Stratified split, Albumentations artırımı, gerçek OCR ağırlıkları ve benchmarklar tamamlandı | ÖLÇÜLDÜ |
| 1D CNN+LSTM Değerlendirmesi | Zamansal sınıflandırıcı ve veri pencereleri tespiti | HEDEF |

Not: “Tamamlandı” ve "ÖLÇÜLDÜ" kod ve yerel otomatik kabulü belirtir; dış servis/T4/Turkcell kanıtı veya henüz entegre edilmemiş gelecek hedefleri (LSTM gibi) kapsamaz.
