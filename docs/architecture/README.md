# Architecture Index

## High-level pipeline

```text
Camera / Video
    ↓
Image Conditioning
    ↓
YOLOv8 Perception
    ↓
Driver-Action + Object Detections
    ↓
Tracking / Temporal State
    ↓
BEV / Homography
    ↓
Timestamp-Aware Motion Estimation
    ↓
Confidence-Adaptive Kalman Filtering
    ↓
Offline FTR Output | Live / 5G Extension
```

## Main architecture references

- `../../5G PROJE/ARCHITECTURE.md` — detailed system architecture.
- `../../5G PROJE/reports/system_architecture_summary.md` — architecture summary.
- `../../5G PROJE/ftr_core/` — offline FTR core.
- `../../5G PROJE/final_5g_extension/` — live/5G extension layer.
- `../../Dockerfile` — official FTR container boundary.

## Architectural principle

The competition-ready offline FTR path and the network-assisted live extension are deliberately separable. This keeps the evaluation container deterministic while allowing connected-system capabilities to evolve independently.

For claim evidence, see [`../evidence/README.md`](../evidence/README.md).
