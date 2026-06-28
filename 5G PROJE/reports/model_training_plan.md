# Model Eğitim Planı — Faz 3

**Proje:** SİNAPTİC5G — FTR Yarışma Sistemi  
**Tarih:** 2026-06-25  
**Üretim Kilidi:** `models/detector.onnx` (detector_v3) KESİNLİKLE KILITLI ✅

---

## Genel Bakış

Bu rapor, 3 deneysel model konfigürasyonu için eğitim planını içerir.
Hiçbir model otomatik olarak production'a alınmaz.
V4+ yalnızca detector_v3 metriklerini **kanıta dayalı** aştığı durumda önerilebilir.

---

## Deney 1: detector_v4_full (YOLOv8m, 50 Epoch)

**Config:** `configs/train_detector_v4_full.yaml`

| Parametre | Değer |
|-----------|-------|
| Model | YOLOv8m |
| Epochs | 50 |
| Batch | 16 (8GB GPU) |
| Optimizer | AdamW (lr=0.001) |
| AMP | ✅ FP16 |
| Freeze | 10 katman |
| Warmup | 3 epoch |
| Early Stopping | patience=15 |

**Eğitim Komutu:**
```bash
# GPU varsa:
yolo train cfg=configs/train_detector_v4_full.yaml

# GPU yoksa dry-run:
yolo train cfg=configs/train_detector_v4_full.yaml epochs=1 batch=2 imgsz=320 device=cpu
```

**Çıktı:** `models/runs/experiments/detector_v4_full_50ep/`

---

## Deney 2: YOLOv8l Büyük Model

**Config:** `configs/train_yolov8l_experiment.yaml`

| Parametre | Değer |
|-----------|-------|
| Model | YOLOv8l (büyük) |
| Epochs | 50 |
| Batch | 8 (12GB GPU) |
| Optimizer | AdamW (lr=0.0005) |
| Freeze | 15 katman |
| Warmup | 5 epoch |

**Motivasyon:** Büyük model → düşük destekli sınıflarda (teknocan, bilgisayar) potansiyel iyileşme.  
**Risk:** CPU latency artışı — FTR budget kontrol zorunlu.

**Eğitim Komutu:**
```bash
yolo train cfg=configs/train_yolov8l_experiment.yaml

# CPU smoke-run:
yolo train cfg=configs/train_yolov8l_experiment.yaml epochs=1 batch=1 imgsz=320 device=cpu
```

**Çıktı:** `models/runs/experiments/yolov8l_experiment/`

---

## Deney 3: Class-Balanced Loss

**Config:** `configs/train_balanced_loss.yaml`

| Parametre | Değer |
|-----------|-------|
| Model | YOLOv8m |
| cls | 0.8 (artırıldı) |
| cls_pw | 1.0 (focal loss approximation) |
| copy_paste | 0.20 (yüksek — düşük destekli sınıflar) |
| rect | false (dengeli sampling) |

**Hedef sınıflar:** `teknocan`, `bilgisayar`, `emniyet_kemeri_ihlali`

**Eğitim Komutu:**
```bash
yolo train cfg=configs/train_balanced_loss.yaml

# CPU smoke-run:
yolo train cfg=configs/train_balanced_loss.yaml epochs=1 batch=2 imgsz=320 device=cpu
```

**Çıktı:** `models/runs/experiments/balanced_loss_experiment/`

---

## Deney 4: detector_v5 (60 Epoch - Sınıf Dengeleme ve Copy-Paste)

**Config:** `configs/train_detector_v5.yaml`

| Parametre | Değer |
|-----------|-------|
| Model | YOLOv8m |
| Epochs | 60 (50 + 10 warmup/fine-tune) |
| Batch | 12 (8GB RTX 4070 optimize) |
| Veri Seti | `data/curated/detector_v5` (oversampled train, augmented val) |
| Sınıf Ağırlıkları | `cls_pw=0.3` (düşük destekli sınıf dengeleme) |
| Optimizer | AdamW (lr0=0.001, lrf=0.01) |
| Freeze | 10 katman |
| AMP | ✅ FP16 |

**Motivasyon:** `teknocan` train 90→723 (copy-paste), `bilgisayar` val 10→50 (augmentation), `emniyet_kemeri` train 654→1260 (copy-paste). Bu şekilde düşük destekli sınıflarda AP50 ve Recall değerlerinin radikal biçimde artırılması hedeflenmiştir.

**Eğitim Komutu:**
```bash
python scripts/train_detector_v5.py

# CPU/GPU smoke-run:
python scripts/train_detector_v5.py --smoke-run
```

**Çıktı:** `models/runs/experiments/detector_v5_60ep/`

---

## CRNN Fine-Tune (Türk Plaka OCR)

**Script:** `scripts/train_crnn_tr_plate.py`  
**Veri:** `data/ocr_test/` (300+ plaka crop — bkz. data/ocr_test/README.md)

```bash
# GPU eğitimi:
python scripts/train_crnn_tr_plate.py \
    --train-dir data/ocr_test/images \
    --label-dir data/ocr_test/labels \
    --output-dir models/runs/crnn_tr_plate \
    --epochs 30 --batch 32 --device cuda

# GPU yok (smoke-run):
python scripts/train_crnn_tr_plate.py --smoke-run
```

**Metrik hedefleri:**
- Exact match rate ≥ 70%
- CER ≤ 10%

---

## CNN-LSTM Temporal Model İyileştirmesi

**Script:** `scripts/train_temporal_lstm.py`  
**Mimari:** BiLSTM + Attention (2 katman, hidden=64, dropout=0.3)

```bash
# GPU eğitimi:
python scripts/train_temporal_lstm.py \
    --output-dir models/runs/cnn_lstm_improved \
    --epochs 50 --batch 64 --device cuda

# CPU dry-run:
python scripts/train_temporal_lstm.py --smoke-run
```

**Sınıflar:** 5 (normal_surus, telefonla_konusma, sigara_icme, uyuklama, esneme)

---

## Karşılaştırma Kriterleri (v4 → Production'a Geçiş Eşiği)

| Kriter | Zorunlu mu? | Eşik |
|--------|-------------|------|
| mAP50 ≥ detector_v3.mAP50 | ✅ ZORUNLU | |
| mAP50-95 ≥ detector_v3 | ✅ ZORUNLU | |
| FTR CPU latency ≤ 8s/video | ✅ ZORUNLU | |
| teknocan recall artışı | Tercih edilen | |
| bilgisayar recall artışı | Tercih edilen | |
| emniyet_kemeri recall artışı | Tercih edilen | |

**Değerlendirme Script:** `scripts/evaluate_detector_v4_full.py` (Faz 4 sonunda)

---

## Üretim Güvenliği

- ✅ `models/detector.onnx` (detector_v3) KESİNLİKLE KILITLI
- ✅ Tüm deney çıktıları `models/runs/experiments/` altında
- ✅ Otomatik production geçişi YOK
- ✅ `model_lock.json` yalnızca `export_and_lock.py` tarafından güncellenir

---

*Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)*
