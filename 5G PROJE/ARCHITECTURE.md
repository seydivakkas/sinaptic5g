# SİNAPTİC5G Kanonik Mimari

## Sınırlar

| Bileşen | Sorumluluk | Kabul etmediği veri |
|---|---|---|
| BFF | Kimlik, QoD, runtime ICE config, 60 sn SDP posta kutusu | RTP/RTCP, video, frame, telemetri |
| Redis | Tek kullanımlık offer/answer ve dağıtık QoD kilidi | Medya, kalıcı kaynak gerçek |
| GPU medya servisi | WebRTC peer, vehicle sentinel, sonuç telemetrisi | Resmi FTR serialization |
| Android | CameraX, TFLite vehicle sentinel, P2P publisher, UI, Room upsert | Client secret, plaka/görüntü/SDP/token saklama |
| FTR container | Çevrimdışı decode, resmi enum/schema serializer | Ağ, QoD, Android, Redis |

## Signaling durumu

1. Android JWT ile runtime ICE ayarını alır.
2. ICE gathering tamamlanınca offer’ı BFF’ye bırakır.
3. GPU servis tokenıyla offer’ı Redis kuyruğundan atomik alır.
4. GPU peer/answer oluşturup BFF’ye bırakır.
5. Android 500 ms aralıkla, en çok 60 saniye answer bekler.
6. Answer tesliminde offer ve answer Lua işlemiyle birlikte silinir.
7. RTP/RTCP doğrudan Android↔GPU akar.

Redis yoksa signaling `503 signaling_unavailable` döner. Android yerel sentinel ile çalışmayı sürdürür. Ayrı ICE candidate endpoint’i, signaling WebSocket’i ve ham medya fallback’i yoktur. SDP UTF-8 olarak en çok 64 KiB’dir.

## Model ve nullable alanlar

`AnalysisResult` ve yarışma enum’ları korunur. Dahili aktif yetenek yalnız `vehicle`dır. Plaka, tip, renk, davranış, kabin, nesne ve yolcu sonuçları kanıtlı yarışma modeli gelene kadar nullable kalır. `unknown` resmi serializer’a yazılmaz.

Manifest model asset hash’ini, tensor şekillerini/türlerini, COCO sınıflarını ve domain mapping’i taşır. `model_lock.json`, build sırasında manifest ile paketlenen byte’ların aynı export’a ait olduğunu doğrular.

## Telemetri ve idempotency

GPU mesajı `event_id`, `frame_id`, capture/receive/complete zamanları, risk, detection listesi, dropped-frame sayısı ve nullable yarışma alanlarını taşır. GPU cihaz başına son envelope’u saklar. Reconnect snapshot’ı aynı `event_id` ile gönderir; Room `REPLACE` upsert uygular.

## Fail-safe davranış

- TFLite eksik/hash veya tensor sözleşmesi bozuk: mobil inference kapalı, uygulama açık.
- Redis yok: signaling 503, yerel sentinel devam.
- ICE/peer hatası: yerel sentinel devam; REST/WebSocket medya fallback’i yok.
- Yavaş GPU: queue büyümez, bekleyen eski kare atılır.
- QoD provider/kanıt yok: inference kesilmez, Best Effort devam eder.
