# SİNAPTİC5G — Proje Mermaid Diyagramları (Teknik Kanıt Paketi)

> **Proje:** SİNAPTİC5G  
> **Tarih:** 2026-06-22  
> **Kaynak doğrulaması:** `ARCHITECTURE.md`, `ftr_main.py`, `media_service.py`, `server.py`, `api/qod_client.py`, `api/qod_orchestrator.py`, `api/vehicle_sentinel.py`, `api/webrtc_signaling.py`, `reports/qod_decision_simulation.md`, `reports/ftr_performance_profile.md`  
> **Kanıt disiplini:** ÖLÇÜLDÜ / TAMAMLANDI / ⚠️ UYARI / 🎯 HEDEF  

---

## DİYAGRAM 1 — Genel Yol Haritası ve Proje Aşamaları

### Açıklama
Projenin 5 ana aşamasını ve bu aşamalar arasındaki girdi/çıktı ilişkisini gösterir. Proje yaşam döngüsünü soldan sağa takip eden bu diyagram, **sunum açılış slaytı** olarak kullanılabilir; jüriye "projemizin hangi aşamada ne yaptık" sorusunu tek bakışta yanıtlar.

```mermaid
flowchart LR
    subgraph A1["Aşama 1 — Veri Hazırlığı & Kalibrasyon ✅"]
        A1a["Açık Kaynak Veri Setleri\n(Roboflow, Kaggle)"]
        A1b["Kanonik Sınıf Sözlüğü\n(9 Sınıf, ID 0–8)"]
        A1c["Video/Oturum Bazlı Split\n+ Sızıntı Kontrolü"]
        A1d["Karantina & Lisans Denetimi\n(CC BY 4.0 / CC0)"]
        A1a --> A1b --> A1c --> A1d
    end

    subgraph A2["Aşama 2 — YZ Modeli Geliştirme & Optimizasyon ✅"]
        A2a["YOLOv8m Fine-Tuning\n(Phase 1: 50 epoch, Phase 2: 10 epoch)"]
        A2b["detector_v3 — Üretim Modeli\nmAP@0.50: 0.9164"]
        A2c["ONNX FP16 Export\n+ SHA-256 Model Lock"]
        A2d["detector_v4 Korpusu\n(14.200 görüntü — Deneysel 🎯)"]
        A2a --> A2b --> A2c
        A2d -. "Deneysel: 2-epoch, mAP: 0.3042" .-> A2b
    end

    subgraph A3["Aşama 3 — 5G API Entegrasyonu & QoD Yönetimi ✅ / ⚠️"]
        A3a["CAMARA QoD API İstemcisi\n(qod_client.py)"]
        A3b["QoD Orkestratör\n(qod_orchestrator.py)"]
        A3c["Durum Makinesi\nIDLE→OBSERVE→ACTIVE→COOLDOWN"]
        A3d["Gerçek CAMARA Testi\n⚠️ UYARI: 5G SIM & Sandbox Gerekli"]
        A3a --> A3b --> A3c -. "Fiziksel SIM yok" .-> A3d
    end

    subgraph A4["Aşama 4 — Edge & MEC Katmanları"]
        subgraph A41["4.1 Edge (Mobil) ✅"]
            A41a["Android CameraX\nVideo Capture"]
            A41b["TFLite Vehicle Sentinel\n(Yerel Ön Filtre)"]
            A41c["WebRTC P2P Yayın\n(aiortc / RTP-SRTP)"]
            A41a --> A41b --> A41c
        end
        subgraph A42["4.2 MEC (Sunucu) ✅"]
            A42a["GPU Media Service\n(media_service.py)"]
            A42b["ONNX Runtime Çıkarım\ndetector_v3 — CPU/GPU EP"]
            A42c["BFF: FastAPI server.py\nRedis Signaling & QoD Lock"]
            A41c -->|"WebRTC RTP/SRTP"| A42a --> A42b --> A42c
        end
    end

    subgraph A5["Aşama 5 — Entegrasyon, Validasyon & Teslim ✅"]
        A5a["Offline FTR Hattı\n(ftr_main.py)"]
        A5b["JSON Schema Doğrulama\n(Draft 2020-12, additionalProperties=false)"]
        A5c["CPU E2E Benchmark\nOrt: 2.24s ≤ 8s ✅"]
        A5d["Pytest 42/42 ✅"]
        A5e["results.json\n(Resmi Teslimat Çıktısı)"]
        A5a --> A5b --> A5e
        A5c --> A5e
        A5d --> A5e
    end

    A1d --> A2a
    A2c --> A5a
    A2c --> A42b
    A3c --> A42c
    A42c --> A5a
```

**Sunumda nasıl kullanılır:** Proje kapsamı bölümünün (1. bölüm) ilk slaytı. "Projemiz kaç aşamadan oluşuyor, her aşamanın çıktısı ne?" sorusuna görsel yanıt verir.

---

## DİYAGRAM 2 — Katmanlı Sistem Mimarisi

### Açıklama
Sistemi 5 mimari katmana ayırarak her katmanın bileşenlerini ve katmanlar arası veri/kontrol akışını gösterir. **Jüriye mimari genel bakış** için kullanılabilir; özellikle "Offline FTR ve Live 5G nasıl ayrışıyor?" sorusunun görsel yanıtıdır.

```mermaid
flowchart TB
    subgraph L1["🗄️ Katman 1 — Veri & Eğitim (Offline)"]
        direction LR
        DS1["driver_distraction_v3\n4.349 görüntü, CC BY 4.0"]
        DS2["cigarette_smokers_v5\n8.391 görüntü, CC BY 4.0"]
        DS3["seatbelt_v1\n500 curated, CC BY 4.0"]
        DS4["kaggle_turkish_plate\n1.946 görüntü, CC0-1.0"]
        DS5["laptop_incar_synthetic\n300 görüntü, Özel"]
        CORP["detector_v3 Korpusu\nTrain:9.662 Val:2.695 Test:1.343\nToplam: 13.700 görüntü"]
        TRAIN["YOLOv8m Fine-Tune\nPhase1: 50 epoch lr=0.01\nPhase2: 10 epoch lr=0.001"]
        ONNX["detector_v3.onnx (FP16)\nSHA-256 Kilitli\nmAP@0.50: 0.9164"]
        DS1 & DS2 & DS3 & DS4 & DS5 --> CORP --> TRAIN --> ONNX
    end

    subgraph L2["📱 Katman 2 — Edge / Mobil (Android)"]
        direction LR
        CAM["CameraX\nVideo Yakalama"]
        TFL["TFLite Vehicle Sentinel\n(Yerel Ön Filtre — Araç Var/Yok)"]
        SDP["SDP Offer Üretimi\n(WebRTC ICE Gathering)"]
        JWT["JWT Auth\n(BFF'den ICE Config)"]
        CAM --> TFL --> SDP
        JWT --> SDP
    end

    subgraph L3["🖥️ Katman 3 — MEC / Sunucu (GPU Media Service)"]
        direction LR
        PEER["WebRTC Peer\n(aiortc — RTP/SRTP)"]
        ZDCE["Zero-DCE\nDüşük Işık Önişleme"]
        INFER["ONNX Runtime\ndetector_v3 Çıkarım\nCPU EP / GPU EP"]
        OCR["LPRNet OCR\n+ Regex Sentinel\nScore = Cbox × Cocr"]
        TRACK["IOU State Tracker\nN=15 kare kararlılık kuralı"]
        RISK["Risk Scorer\n(risk_scorer.py)"]
        COLOR["HSV Araç Rengi\nROI Histogram"]
        PEER --> ZDCE --> INFER --> OCR & TRACK & COLOR --> RISK
    end

    subgraph L4["📡 Katman 4 — 5G API & QoD Yönetimi"]
        direction LR
        BFF["FastAPI BFF\n(server.py)\nREST + WebSocket"]
        REDIS["Redis\n• SDP Offer/Answer (60s TTL)\n• QoD Distributed Lock\n• Ham Video SAKLANMAZ"]
        QOD["QoD Orkestratör\n(qod_orchestrator.py)\nIDLE→OBSERVE→ACTIVE→COOLDOWN"]
        CAMARA["CAMARA QoD Client\n(qod_client.py)\nPOST /qod-sessions"]
        NUM["CAMARA Number\nVerification\n(number_verification.py)"]
        BFF <--> REDIS
        BFF --> QOD --> CAMARA
        BFF --> NUM
    end

    subgraph L5["✅ Katman 5 — FTR / Validasyon & Teslim"]
        direction LR
        FTR["ftr_main.py\nÇevrimdışı FTR Hattı\n(Redis, Ağ, 5G SIM YOK)"]
        SCHEMA["JSON Schema\nDraft 2020-12\nadditionalProperties=false"]
        BENCH["CPU E2E Benchmark\nOrt: 2.24s / Min: 2.24s\nMax: 2.25s — UYUMLU ✅"]
        PYTEST["pytest tests/\n42/42 BAŞARILI ✅"]
        RESULT["results.json\n(Resmi FTR Çıktısı)"]
        FTR --> SCHEMA --> RESULT
        BENCH & PYTEST --> RESULT
    end

    ONNX -->|"Model Kopyası"| L3
    ONNX -->|"FTR Çıkarım"| FTR
    SDP -->|"WebRTC RTP/SRTP\n(Ham video BFF/Redis'e GELMİYOR)"| PEER
    RISK -->|"Detection JSON\n~150 byte/kare"| BFF
    QOD -.->|"QOS_L_PRIORITY\n⚠️ Fiziksel SIM gerekli"| CAMARA
```

**Sunumda nasıl kullanılır:** Çözüm mimarisi bölümünün (3.2) ana slaytı. Her katmanı ayrı ayrı göstererek "hangi bileşen nerede çalışıyor?" sorusunu yanıtlar.

---

## DİYAGRAM 3 — Uçtan Uca Veri Akışı (Offline FTR + Live 5G Dalları)

### Açıklama
Ham verinin kaynaktan final `results.json` çıktısına kadar geçtiği her adımı gösterir. Offline FTR ve Live 5G kolları ayrı dallar olarak çizilmiştir. **Teknik sınama bölümünde** (4. bölüm) "sistemi nasıl test ettiniz?" sorusuna görsel yanıt verir.

```mermaid
flowchart TD
    START(["🎬 Ham Veri Kaynağı\n(Kamera / Video Dosyası)"])

    subgraph PREP["Aşama 1 — Veri Hazırlığı & Etiketleme"]
        P1["Roboflow / Kaggle İndirme\n+ Lisans Doğrulama"]
        P2["YOLO Format Etiket\n+ Label Remap (9 sınıf)"]
        P3["Karantina Filtresi\n(soft_toy → quarantine)"]
        P4["Video/Oturum Bazlı Split\nassert_no_group_leakage ✅"]
        P5["SHA-256 Manifesto\n+ Corpus Lock"]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    subgraph TRAIN_BLOCK["Aşama 2 — Eğitim & Optimizasyon"]
        T1["YOLOv8m Phase 1\n50 epoch, lr=0.01, mosaic=1.0"]
        T2["YOLOv8m Phase 2\n10 epoch, lr=0.001, backbone freeze"]
        T3["detector_v3.pt → ONNX FP16\nSHA-256 Lock: c47be5a00..."]
        T4["detector_v4 Korpusu\n14.200 görüntü, seatbelt_v1 +350"]
        T5["detector_v4_full 2-epoch\nmAP: 0.3042 — KEEP detector_v3"]
        T1 --> T2 --> T3
        T4 -. "Deneysel 🎯" .-> T5
        T5 -. "Baseline'ı geçemedi\nDO NOT PROMOTE" .-> T3
    end

    START --> PREP
    P5 --> T1

    T3 -->|"Üretim Modeli"| OFFLINE
    T3 -->|"Canlı Çıkarım Modeli"| LIVE

    subgraph OFFLINE["Çevrimdışı FTR Hattı ✅"]
        direction TB
        O1["OpenCV Video Capture\n(ftr_main.py)"]
        O2["Adaptive Stride\nDinamik Kare Atlama"]
        O3["Zero-DCE\nDüşük Işık İyileştirme\n~4.78ms/kare"]
        O4["ONNX Runtime CPU\ndetector_v3 Çıkarım\n~965ms/kare (darboğaz)"]
        O5["NMS + Koordinat Ölçekleme\nBBox & Sınıf Tensörleri"]
        O6["Plaka Crop → LPRNet OCR\nScore = Cbox × Cocr ≥ 0.70\nRegex: Türk Plaka Standardı"]
        O7["IOU State Tracker\nN=15 kare kararlılık kuralı"]
        O8["FTR State Aggregator\nSüre, Olay Sayısı, Risk Skoru"]
        O9["JSON Writer\nresults.json Draft 2020-12"]
        O10["jsonschema Doğrulama\nadditionalProperties=false ✅"]
        O1 --> O2 --> O3 --> O4 --> O5
        O5 --> O6 --> O8
        O5 --> O7 --> O8
        O8 --> O9 --> O10
    end

    subgraph LIVE["Canlı 5G / WebRTC Hattı ✅ / ⚠️"]
        direction TB
        L1["Android CameraX\nH.264 Video Encode"]
        L2["TFLite Vehicle Sentinel\n(Edge Ön Filtre)"]
        L3["JWT Auth → BFF\nICE Config Alma"]
        L4["SDP Offer → BFF\n(Redis 60s TTL)"]
        L5["GPU Media Service Peer\n(aiortc WebRTC)"]
        L6["RTP/SRTP Akışı\n(Ham Video BFF/Redis'e GELMİYOR)"]
        L7["GPU ONNX Runtime\ndetector_v3 Çıkarım"]
        L8["Detection JSON Telemetrisi\n~150 byte/kare → Redis"]
        L9["QoD Orkestratör\nFayda Eşiği: 0.65"]
        L10["CAMARA QoD Client\nPOST /qod-sessions\n⚠️ Fiziksel SIM Gerekli"]
        L11["5G Network Slice\nQOS_L_PRIORITY"]
        L12["Redis Fail-Safe\n503 → Yerel Sentinel Devam"]
        L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7
        L7 --> L8 --> L9 --> L10 --> L11
        L4 -. "Redis yok" .-> L12
        L10 -. "Provider yok\nBest Effort" .-> L12
    end

    subgraph VALID["Aşama 5 — Validasyon & Teslim"]
        V1["CPU E2E Benchmark\n5x iterasyon\nOrt: 2.24s ✅"]
        V2["pytest tests/\n42/42 ✅\nBFF, WebRTC, QoD, Manifest"]
        V3["Hata Analizi\nFP:43, FN:59 / 150 görüntü"]
        V4["Robustness Stres Testi\n8 bozulma senaryosu"]
        V5["Eşik Kalibrasyonu\nSınıf 5 optimal: 0.90"]
        V6["FTR Performans Profili\nDarboğaz: YOLO ~965ms"]
        FINAL(["📦 results.json\nResmi Teslimat"])
    end

    O10 --> VALID
    L8 --> VALID
    V1 & V2 & V3 & V4 & V5 & V6 --> FINAL
```

**Sunumda nasıl kullanılır:** Çözümün sınanması bölümünün (4. bölüm) açılış slaytı. "Veri nereden geliyor, model nasıl çıkarım yapıyor, sonuç nasıl üretiliyor?" sorularını tek akışta yanıtlar.

---

## DİYAGRAM 4 — Entegrasyon, Bileşen Statüsü ve Final Üretim Kararı

### Açıklama
Tüm bileşenleri **üretimde aktif / deneysel / açık hedef** statülerine göre ayıran karar diyagramı. **Jüri ve danışmana "hangi bileşen hazır, hangisi neden hazır değil?" sorusunun dürüst ve şeffaf yanıtıdır.** Teknik sınama bölümünün kapanış slaytı olarak kullanılabilir.

```mermaid
flowchart TB
    subgraph PROD["🟢 ÜRETİMDE AKTİF — TAMAMLANDI / ÖLÇÜLDÜ"]
        direction LR
        P_MODEL["detector_v3.onnx\nmAP@0.50: 0.9164\nPrecision: 0.9225 / Recall: 0.8905"]
        P_FTR["ftr_main.py\nOffline FTR Hattı\nJSON Schema ✅"]
        P_BENCH["CPU E2E: 2.24s\nHedef ≤8s — UYUMLU ✅"]
        P_PYTEST["pytest 42/42 ✅"]
        P_SCHEMA["results.json\nDraft 2020-12 Şema Onaylı ✅"]
        P_REDIS["Redis Signaling\nSDP TTL 60s / Atomic Offer-Answer ✅"]
        P_WEBRTC["WebRTC P2P\nRTP/SRTP Ham Video Zero-Storage ✅"]
        P_QOD_SIM["QoD Durum Makinesi Simülasyonu\n7 Senaryo Doğrulandı ✅"]
        P_TRACKER["IOU State Tracker\nN=15 Kare Kararlılık ✅"]
        P_OCR_PIPE["OCR Pipeline (LPRNet)\nLatency Ölçüldü ✅"]
        P_HSV["HSV Araç Rengi\nROI Histogram ✅"]
        P_ZDCE["Zero-DCE Önişleme\nOrtalama 4.78ms ✅"]
        P_HASHLOCK["Model Hash Lock\nSHA-256 Doğrulama ✅"]
    end

    subgraph EXP["🟡 DENEYSEL — TAMAMLANDI, ÜRETİME ALINMADI"]
        E_V4CORP["detector_v4 Korpusu\n14.200 görüntü, seatbelt_v1 +350\nSHA-256 Kilitli ✅"]
        E_V4FULL["detector_v4_full\n2-epoch, mAP@0.50: 0.3042\nKısıtlı GPU — KEEP detector_v3"]
        E_V4SMOKE["detector_v4_smoke\n1-epoch %5 fraksiyon\nPipeline Doğrulama Amaçlı"]
        E_ERR["Hata Analizi Galerisi\nFP/FN Dağılımı ✅"]
        E_ROB["Robustness Stres Testi\n8 Bozulma Senaryosu ✅"]
        E_THRESH["Eşik Kalibrasyonu\nSınıf 5 optimal: 0.90 ✅"]
        E_PROFILE["FTR Performans Profili\nBileşen Bazlı ms ✅"]
    end

    subgraph WARN["🔴 AÇIK HEDEF / UYARI — DIŞ BAĞIMLILIK"]
        W_TEKNOCAN["Teknocan Sentetik Veri\n⚠️ Ön Plan PNG Eksik\nScript hazır: generate_teknocan_synthetic.py"]
        W_OCR_ACC["OCR CER & Exact-Match\n🎯 HEDEF: Ground-truth Plaka Seti Yok\nDeğerlendirme kiti hazır"]
        W_CAMARA["Gerçek Turkcell CAMARA Testi\n⚠️ Fiziksel 5G SIM + Sandbox Gerekli\nSimülasyon doğrulandı"]
        W_V4FULL["detector_v4 Tam Eğitim\n🎯 HEDEF: 50-epoch GPU Ortamı Gerekli\nKorpus hazır"]
        W_T4["CUDA/T4 Benchmark\n🎯 HEDEF: GPU Ortamı Gerekli\nCPU uyumlu, T4 beklenen"]
    end

    subgraph DECISION["⚖️ Final Model Kararı"]
        DEC["KEEP detector_v3\nmAP@0.50: 0.9164\nDetector_v4 → DO NOT PROMOTE"]
    end

    P_MODEL --> DECISION
    E_V4FULL -. "Baseline'ı geçemedi" .-> DECISION
    DECISION --> P_FTR
    DECISION --> P_WEBRTC

    PROD --> FINAL_OUTPUT(["📦 Teslim Paketi\nresults.json + pytest 42/42\n+ CPU 2.24s + SHA-256 Kilitli Model"])
    EXP --> EVIDENCE(["📊 Kanıt Paketi\n73 rapor dosyası\nFINAL_EVIDENCE_INDEX.md"])
    WARN --> ROADMAP(["🗺️ Gelecek Çalışmalar\nHEDEF/UYARI Raporu\nfinal_quality_gate.md"])
```

**Sunumda nasıl kullanılır:** Projenin kapanış slaytı. "Projede ne hazır, ne deneysel, ne eksik?" sorusunu renk koduyla (yeşil/sarı/kırmızı) şeffaf biçimde yanıtlar; jürinin dürüstlük değerlendirmesini güçlendirir.

---

## EK — QoD Durum Makinesi Diyagramı (Doğrulama Kanıtı)

### Açıklama
`reports/qod_decision_simulation.md` dosyasındaki 7 simülasyon senaryosundan üretilmiştir. **Canlı 5G entegrasyon bölümünde** doğrulama kanıtı olarak sunulabilir.

```mermaid
stateDiagram-v2
    [*] --> IDLE : Sistem Başlatıldı

    IDLE --> OBSERVE : Araç/Sürücü Sinyali Alındı
    OBSERVE --> IDLE : Hedef Yok / S1 RED

    OBSERVE --> OBSERVE : Hedef Sabit / Fayda Düşük\nS2 RED: target_not_approaching\nS3 RED: calibrated_gate (Fayda=0.15 < 0.65)

    OBSERVE --> REQUESTING : Fayda Eşiği Aşıldı\nS4 KABUL: Fayda=0.860 > 0.65

    REQUESTING --> ACTIVE : QoD Oturumu Sağlandı\nCONNECTED → QOS_L_PRIORITY\nSession: mock-session-1

    REQUESTING --> FAILED : CAMARA Provider Hatası\n(⚠️ Fiziksel SIM Yok)

    ACTIVE --> ACTIVE : Duplikasyon Engellendi\nS5 RED: equivalent_session_active

    ACTIVE --> COOLDOWN : Oturum Kapatıldı\nS6 KABUL: stopped → Best Effort

    FAILED --> COOLDOWN : Hata Cooldown Başladı\nInference Kesilmez

    COOLDOWN --> COOLDOWN : S7 RED: cooldown_active\nErken Talep Engellendi

    COOLDOWN --> IDLE : Cooldown Süresi Bitti\nSisteme Hazır

    note right of ACTIVE
        Inference kesilmez.
        Best Effort devam eder.
    end note

    note right of FAILED
        Redis yok → 503
        Yerel Sentinel devam eder.
    end note
```

**Sunumda nasıl kullanılır:** "5G CAMARA testini gerçek ortamda yapmadınız, sistemi nasıl doğruladınız?" sorusuna yanıt. 7 simülasyon senaryosunun her birinin bu diyagramda hangi geçişe karşılık geldiği gösterilebilir.

---

*Bu belge SİNAPTİC5G projesinin jüri sunumu için hazırlanmıştır.*  
*Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas) — Tüm Hakları Saklıdır.*
