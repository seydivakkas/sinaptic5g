# SİNAPTİC5G — detector_v5 Training Log

> **Model:** YOLOv8m (Candidate v5)
> **Mode:** Smoke-Run

## 1. Eğitim Ortamı (Environment)

* Python Version: 3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]
* PyTorch Version: 2.11.0+cu126
* Ultralytics Version: 8.4.41
* ONNX Runtime Version: 1.27.0
* CUDA Available: True
* CUDA Version: 12.6
* GPU Name: NVIDIA GeForce RTX 4070 Laptop GPU
* GPU VRAM: 8.00 GB

## 2. Eğitim Konfigürasyonu (Recipe)

```yaml
# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
#
# Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
# amaçlı olarak paylaşılmıştır.
#
# YASAKLAR:
#   1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
#   2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
#   3. Alt lisanslanamaz, satılamaz veya devredilemez.
#   4. Tersine mühendislik yapılamaz.
#
# İZİN VERİLEN KULLANIM:
#   - GitHub üzerinde görüntüleme ve okuma.
#   - Kişisel öğrenim amacıyla kodu inceleme (kopyalamadan).
#
# YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
# İzin talepleri için: GitHub @seydivakkas

# detector_v5 Eğitim Konfigürasyonu (60 Epoch)
# Kullanım:
#   yolo train cfg=configs/train_detector_v5.yaml
#   # Veya Python scripti ile:
#   python scripts/train_detector_v5.py

# ─── Model ───────────────────────────────────────────────────────────────────
model: yolov8m.pt          # YOLOv8-medium baseline model

# ─── Veri Seti ───────────────────────────────────────────────────────────────
data: configs/data_v5.yaml  # detector_v5 veri seti YAML

# ─── Eğitim Parametreleri ────────────────────────────────────────────────────
epochs: 60
batch: 12                  # 8GB VRAM RTX 4070 için optimize edilmiş
imgsz: 640
device: 0                  # GPU: "0" | CPU: "cpu"
workers: 4
seed: 42
deterministic: true
patience: 20               # Erken durdurma sabrı

# ─── Optimizer ───────────────────────────────────────────────────────────────
optimizer: AdamW
lr0: 0.001                 # Başlangıç LR
lrf: 0.01                  # Bitiş LR = lr0 * lrf
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# ─── Mixed Precision ─────────────────────────────────────────────────────────
amp: true                  # FP16 AMP - GPU hızlandırma

# ─── Çıktı Yolları ───────────────────────────────────────────────────────────
project: models/runs/experiments
name: detector_v5_60ep
save: true
save_period: 10            # 10 epoch'ta bir checkpoint kaydet

# ─── Freeze (Transfer Learning) ──────────────────────────────────────────────
freeze: 10                 # Backbone transfer öğrenimi için ilk 10 katmanı dondur

# ─── Augmentasyon (YOLOv8 Dahili) ────────────────────────────────────────────
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10.0
translate: 0.1
scale: 0.5
shear: 2.0
perspective: 0.0005
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.15
copy_paste: 0.15
close_mosaic: 10

# ─── Kayıp Fonksiyonu ────────────────────────────────────────────────────────
box: 7.5
cls: 0.5
dfl: 1.5
cls_pw: 0.3                # Sınıf ağırlık üssü (düşük destekli sınıf dengeleme)
```

## 3. Eğitim İlerleme Kaydı (Training Progress)

Eğitim başlatıldı...
