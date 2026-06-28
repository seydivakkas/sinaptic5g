# SiNAPTiC5G - detector_v5 E2E Gecikme Benchmark Raporu

> **Tarih:** 2026-06-25 08:31
> **Model:** detector_v5 (YOLOv8m ONNX)
> **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU
> **CUDA Destegi:** Mevcut

## 1. Mod Bazli Performans Tablosu

| Mod | Ortalama Sure | En Hizli | En Yavas | Hedef | Uyumluluk |
|---|:---:|:---:|:---:|:---:|:---:|
| **GPU (CUDA)** | **1.5233s** | 1.5074s | 1.5498s | 8.0s | GEÇTİ ✅ |
| **CPU** | **2.2433s** | 2.2401s | 2.2484s | 8.0s | GEÇTİ ✅ |

## 2. Iterasyon Detaylari

| Iterasyon | GPU (CUDA) Sure | CPU Sure |
|:---------:|:---------------:|:--------:|
| 1 | 1.5082s | 2.2484s |
| 2 | 1.5359s | 2.2421s |
| 3 | 1.5155s | 2.2445s |
| 4 | 1.5074s | 2.2416s |
| 5 | 1.5498s | 2.2401s |

## 3. Degerlendirme ve Analiz

1. **GPU (CUDA) Performansi:**
   - GPU modunda ortalama **1.52s** E2E sure ile 8.0s hedefini basariyla karsilamaktadir. ✅

2. **CPU Performansi:**
   - CPU modunda ortalama **2.24s** E2E sure olculmustur.
   - CPU'da uyarlanabilir kare tarama (adaptive stride = 1.5s) ve azaltilmis warmup sayesinde **8.0s** hedefini basariyla karsilamaktadir. ✅

---
OZEL LiSANS - TUM HAKLAR SAKLIDIR
Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
