"""COCO vehicle-only sentinel used by the GPU WebRTC service.

The output contract deliberately maps only COCO car/bus/truck to the existing
``vehicle`` domain enum. Every other competition field remains nullable.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any


class VehicleSentinel:
    def __init__(self, model_path: str, manifest_path: str) -> None:
        import onnxruntime as ort

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.labels: list[str] = manifest["labels"]
        self.domain_mapping = {
            int(class_id): domain
            for class_id, domain in manifest["domain_mapping"].items()
        }
        if set(self.domain_mapping) != {2, 5, 7} or set(self.domain_mapping.values()) != {"vehicle"}:
            raise ValueError("manifest must activate only COCO car/bus/truck as vehicle")
        providers = [
            provider
            for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        if list(self.input_shape) != [1, 3, 640, 640]:
            raise ValueError(f"unexpected ONNX input shape: {self.input_shape}")
        if not (
            len(self.output_shape) == 3
            and 84 in (self.output_shape[1], self.output_shape[2])
        ):
            raise ValueError(f"unexpected YOLO tensor contract: {self.output_shape}")

        # Warmup execution to compile TensorRT engine/kernels on startup
        try:
            import numpy as np
            dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
            for _ in range(10):
                self.session.run(None, {self.input_name: dummy_input})
        except Exception:
            pass

    def infer(self, bgr: Any, confidence_threshold: float = 0.35) -> tuple[list[dict], float]:
        import cv2
        import numpy as np

        started = perf_counter()
        height, width = bgr.shape[:2]
        scale = min(640 / width, 640 / height)
        resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(bgr, (resized_w, resized_h))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (640 - resized_w) // 2, (640 - resized_h) // 2
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: tensor[None]})[0]
        predictions = output[0]
        if predictions.shape[0] == 84:
            predictions = predictions.T

        boxes: list[list[int]] = []
        scores: list[float] = []
        raw_ids: list[int] = []
        for row in predictions:
            all_scores = row[4:4 + len(self.labels)]
            raw_id = int(np.argmax(all_scores))
            score = float(all_scores[raw_id])
            if score < confidence_threshold or raw_id not in self.domain_mapping:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            x = int((cx - bw / 2 - pad_x) / scale)
            y = int((cy - bh / 2 - pad_y) / scale)
            boxes.append([x, y, int(bw / scale), int(bh / scale)])
            scores.append(score)
            raw_ids.append(raw_id)

        keep = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, 0.45)
        detections = []
        for index in keep:
            index = int(index)
            x, y, box_w, box_h = boxes[index]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + box_w), min(height, y + box_h)
            detections.append({
                "class_id": 5,
                "class_name": "vehicle",
                "source_class_id": raw_ids[index],
                "source_class_name": self.labels[raw_ids[index]],
                "confidence": round(scores[index], 5),
                "bbox": {
                    "x1": x1 / width,
                    "y1": y1 / height,
                    "x2": x2 / width,
                    "y2": y2 / height,
                },
            })
        return detections, (perf_counter() - started) * 1_000
