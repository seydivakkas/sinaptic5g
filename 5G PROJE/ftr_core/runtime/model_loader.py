# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("sinaptic5g.ftr.model_loader")

class ModelLoader:
    """Verifies SHA-256 integrity and initializes onnxruntime sessions securely."""
    
    def __init__(self,
                 lock_path: Path,
                 model_registry: Dict[str, Any]):
        self.lock_path = Path(lock_path)
        self.model_registry = model_registry
        self.lock_data = self._load_lock()
        self.verify_all()

    def _load_lock(self) -> Dict[str, str]:
        if not self.lock_path.is_file():
            # Try root fallback
            alt_path = self.lock_path.parent.parent / self.lock_path.name
            if alt_path.is_file():
                self.lock_path = alt_path
            else:
                raise FileNotFoundError(f"Model lock registry is missing: {self.lock_path}")
                
        with open(self.lock_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def verify_all(self):
        """Verifies actual file hashes against expected locked hashes."""
        logger.info("Starting model lock validation...")
        
        # Required evaluation models
        model_keys = {
            "detector": "detector_optimized_onnx_sha256",
            "coco": "coco_onnx_sha256",
            "lprnet": "lprnet_onnx_sha256",
            "crnn": "crnn_onnx_sha256"
        }
        
        for key, lock_key in model_keys.items():
            model_info = self.model_registry.get(key)
            if not model_info:
                continue
                
            model_path = Path(model_info["path"])
            # Resolve relative to root if needed
            if not model_path.is_file():
                model_path = Path(__file__).resolve().parent.parent.parent / model_path
                
            if not model_path.is_file():
                # Try fallback in root for coco
                if key == "coco":
                    model_path = Path(__file__).resolve().parent.parent.parent / "yolov8n.onnx"
                else:
                    raise FileNotFoundError(f"Locked model file does not exist: {model_path}")
            
            expected_hash = self.lock_data.get(lock_key)
            if not expected_hash:
                # Try fallback to detector_onnx_sha256 if detector_optimized not present in lock
                if key == "detector":
                    expected_hash = self.lock_data.get("detector_onnx_sha256")
                    
            if not expected_hash:
                raise ValueError(f"No hash found in model lock for: {key}")
                
            actual_hash = self._get_sha256(model_path)
            if actual_hash.lower() != expected_hash.lower():
                raise ValueError(f"Integrity violation detected! Model={key}, Expected={expected_hash}, Actual={actual_hash}")
                
            logger.info("Model '%s' hash OK: %s", key, actual_hash[:16])

    def create_session(self, model_key: str) -> Any:
        """Loads and returns an onnxruntime InferenceSession with optimised options."""
        import onnxruntime as ort
        
        model_info = self.model_registry.get(model_key)
        if not model_info:
            raise ValueError(f"No registry configuration found for model key: {model_key}")
            
        model_path = Path(model_info["path"])
        if not model_path.is_file():
            model_path = Path(__file__).resolve().parent.parent.parent / model_path
        if not model_path.is_file() and model_key == "coco":
            model_path = Path(__file__).resolve().parent.parent.parent / "yolov8n.onnx"
            
        available = ort.get_available_providers()
        providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available]
        has_gpu = any(p in ("CUDAExecutionProvider", "TensorrtExecutionProvider") for p in providers)
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4 if has_gpu else 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
        logger.info("Loaded ONNX Session '%s' with providers=%s", model_key, session.get_providers())
        return session
