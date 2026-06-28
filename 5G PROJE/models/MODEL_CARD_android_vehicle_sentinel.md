# Android Vehicle Sentinel Model Card

- Purpose: COCO `car`, `bus` and `truck` detection for local approach gating only.
- It is not a competition task model and is not evidence for plate, type, color or behavior KPIs.
- Source SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
- Artifact SHA-256: `a5b9e6133a5e9af3357dd49bfd059408acaeabcf37f4e31ab3914c8dead1f501`
- Export: `yolo export model=yolov8n.pt format=tflite half=True`
- Active domain capability: `vehicle`; all other domain fields remain nullable.
- Accuracy and device latency status: `HEDEF` until independent measurements exist.
- INT8 and Zero-DCE are blocked until independent calibration and paired-benefit evidence exist.
