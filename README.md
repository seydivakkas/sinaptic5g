<div align="center">

# SİNAPTİC5G

### Real-Time Edge AI & 5G Road-Safety Perception

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer%20Vision-111111?style=flat-square)
![Edge AI](https://img.shields.io/badge/Edge%20AI-Real--Time-0A66C2?style=flat-square)
![5G](https://img.shields.io/badge/5G-Connected%20Intelligence-6F42C1?style=flat-square)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red?style=flat-square)

**An end-to-end intelligent road-safety perception system combining driver-action recognition, object detection, tracking, BEV geometry and robust real-time inference.**

**15,487 governed images** · **9 canonical labels** · **Group-aware split** · **YOLOv8 + Tracking + BEV** · **Adaptive Kalman filtering**

`Computer Vision` · `Edge AI` · `5G` · `YOLOv8` · `Tracking` · `BEV` · `Kalman Filtering` · `Data Governance`

</div>

---

## System objective

SİNAPTİC5G is designed as a **real-time road-safety intelligence pipeline** rather than a standalone detector. It combines perception, temporal tracking and physically meaningful motion reasoning in a deployment-oriented architecture.

The system focuses on two complementary views:

- **Driver / cabin intelligence** — actions and safety violations such as phone use, drinking, looking backward, yawning, smoking and seatbelt violations.
- **Vehicle / road intelligence** — object perception, license-plate detection, tracking, BEV projection and timestamp-aware speed estimation.

---

## Engineering evidence

| Signal | Repository evidence |
|---|---|
| Governed dataset | **15,487 images** |
| Canonical contract | **9 labels** unified across heterogeneous sources |
| Split policy | Group-aware train / validation / test assignment to reduce leakage |
| Class balancing | Downsampling + copy-paste oversampling + class-weight tuning |
| Robustness | Motion blur, Gaussian noise, JPEG compression and illumination variation |
| Motion reasoning | BEV geometry + timestamp-based speed estimation |
| Temporal filtering | Confidence-adaptive Kalman measurement noise |

---

## Perception stack

```text
Camera / Video Input
        ↓
Image Conditioning
  ├── HSV / CLAHE
  └── conditional low-light enhancement
        ↓
YOLOv8 Perception
        ↓
Object / Driver-Action Detections
        ↓
Tracking + Temporal State
        ↓
BEV Projection
        ↓
Timestamp-Aware Motion Estimation
        ↓
Adaptive Kalman Filtering
        ↓
Road-Safety Intelligence Output
```

---

## Dataset governance

Multiple public datasets are normalized into a single **canonical label contract**. The pipeline records source/license information and applies group-aware splitting so frames from the same identity, session or video sequence are not naively distributed across train and evaluation sets.

Current canonical classes include:

`telefonla_konusma` · `su_icme` · `arkaya_bakma` · `esneme` · `sigara_icme` · `emniyet_kemeri_ihlali` · `teknocan` · `bilgisayar` · `license_plate`

---

## Robustness-oriented design

The project explicitly addresses conditions that break naive perception systems:

- illumination changes and glare,
- low-light scenes,
- motion blur,
- camera noise,
- JPEG degradation,
- temporary occlusion,
- dropped / irregular frames,
- and confidence variation across detections.

For motion estimation, time is handled through real timestamps rather than assuming a perfectly fixed frame rate.

---

## Technology stack

**Perception**  
`Python` · `YOLOv8` · `OpenCV` · `Albumentations`

**Tracking & geometry**  
`BEV / Homography` · `Kalman Filtering` · `Timestamp-Based Motion Estimation`

**System engineering**  
`Edge AI` · `5G Integration` · `Data Governance` · `Real-Time Inference`

---

## Documentation

The root README is intentionally portfolio-oriented. The original system documentation is preserved in full:

### **[Full Technical Documentation →](docs/README_FULL.md)**

It contains the detailed dataset construction, canonical mapping, augmentation strategy, perception pipeline, physical reasoning and evaluation documentation.

---

## Engineering principles

1. **Data leakage prevention before benchmark claims**
2. **Real timestamps before frame-count assumptions**
3. **Confidence-aware filtering before fixed-noise tracking**
4. **Physical geometry before purely visual heuristics**
5. **End-to-end system evidence before isolated model metrics**

---

<div align="center">

**Edge perception · temporal reasoning · connected road safety**

[GitHub Profile](https://github.com/seydivakkas) · [Full Documentation](docs/README_FULL.md)

</div>
