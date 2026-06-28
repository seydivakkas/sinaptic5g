# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
#
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

"""
scripts/benchmark_t4_cuda.py — T4 CUDA Benchmark Scripti
=========================================================
Faz 5: Test & Doğrulama

NVIDIA T4 GPU üzerinde model inference latency benchmark'ı yapar.
GPU yoksa CPU benchmark'ı çalışır (dry-run modu).

Ölçülen metrikler:
- Ortalama inference süresi (ms)
- P50, P90, P99 latency
- FPS (frame per second)
- GPU VRAM kullanımı (CUDA varsa)

Hedef modeller:
- models/detector.onnx (üretim — salt okunur)
- models/crnn.onnx (OCR)
- models/cnn_lstm.onnx (temporal)

Kullanım:
    # T4 GPU benchmark:
    python scripts/benchmark_t4_cuda.py --all-models --n-runs 200

    # Sadece detector (CPU):
    python scripts/benchmark_t4_cuda.py \
        --model-path models/detector.onnx \
        --n-runs 100 --batch 1

    # Dry-run (GPU yok):
    python scripts/benchmark_t4_cuda.py --dry-run --n-runs 5

Çıktı formatı: reports/benchmark_t4_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("sinaptic5g.benchmark_t4")

PROJECT_ROOT = Path(__file__).parent.parent

# ─── Benchmark Modelleri ─────────────────────────────────────────────────────
BENCHMARK_MODELS: Dict[str, Dict] = {
    "detector": {
        "path": "models/detector.onnx",
        "input_shape": [1, 3, 640, 640],
        "dtype": "float32",
        "description": "FTR Detector v3 (üretim — salt okunur)",
        "target_latency_ms": 100.0,  # T4 hedef
    },
    "crnn": {
        "path": "models/crnn.onnx",
        "input_shape": [1, 1, 32, 100],
        "dtype": "float32",
        "description": "CRNN Plaka OCR",
        "target_latency_ms": 50.0,
    },
    "cnn_lstm": {
        "path": "models/cnn_lstm.onnx",
        "input_shape": [1, 16, 7],
        "dtype": "float32",
        "description": "CNN-LSTM Temporal Sınıflandırıcı",
        "target_latency_ms": 10.0,
    },
}

# ─── T4 Referans Değerleri ───────────────────────────────────────────────────
T4_REFERENCE = {
    "gpu_memory_gb": 16.0,
    "fp32_tflops": 8.1,
    "fp16_tflops": 65.0,
    "bandwidth_gbs": 300.0,
}


def get_available_providers() -> List[str]:
    """Kullanılabilir ONNX Runtime sağlayıcılarını listeler."""
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except ImportError:
        return []


def get_gpu_info() -> Dict:
    """GPU bilgisini toplar (CUDA varsa)."""
    info = {"available": False}
    
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                info = {
                    "available": True,
                    "name": parts[0] if len(parts) > 0 else "Unknown",
                    "memory_total_mb": int(parts[1]) if len(parts) > 1 else 0,
                    "memory_used_mb": int(parts[2]) if len(parts) > 2 else 0,
                    "temperature_c": int(parts[3]) if len(parts) > 3 else 0,
                    "utilization_pct": int(parts[4]) if len(parts) > 4 else 0,
                }
                info["is_t4"] = "T4" in info.get("name", "")
    except Exception:
        pass
    
    return info


def warmup_session(session, input_name: str, dummy_input: np.ndarray, n_warmup: int = 10) -> None:
    """Session ısınma çalıştırması."""
    for _ in range(n_warmup):
        try:
            session.run(None, {input_name: dummy_input})
        except Exception:
            break


def benchmark_model(
    model_path: Path,
    input_shape: List[int],
    dtype: str = "float32",
    n_runs: int = 100,
    n_warmup: int = 20,
    providers: Optional[List[str]] = None,
) -> Dict:
    """Tek bir model için benchmark çalıştırır."""
    result = {
        "model_path": str(model_path),
        "input_shape": input_shape,
        "n_runs": n_runs,
        "status": "NOT_STARTED",
    }
    
    if not model_path.is_file():
        result["status"] = "MODEL_NOT_FOUND"
        return result
    
    try:
        import onnxruntime as ort
    except ImportError:
        result["status"] = "ONNXRUNTIME_NOT_AVAILABLE"
        return result
    
    if providers is None:
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4
        
        session = ort.InferenceSession(str(model_path), sess_opts, providers=providers)
        input_name = session.get_inputs()[0].name
        actual_providers = session.get_providers()
        
        result["providers"] = actual_providers
        result["using_gpu"] = any("CUDA" in p or "TensorRT" in p for p in actual_providers)
    except Exception as e:
        result["status"] = "LOAD_FAILED"
        result["error"] = str(e)
        return result
    
    np_dtype = np.float16 if dtype == "float16" else np.float32
    dummy_input = np.random.randn(*input_shape).astype(np_dtype)
    
    # Warmup
    LOG.info("  Warmup (%d iter)...", n_warmup)
    warmup_session(session, input_name, dummy_input, n_warmup)
    
    # Benchmark
    LOG.info("  Benchmark (%d iter)...", n_runs)
    latencies_ms: List[float] = []
    
    for i in range(n_runs):
        t0 = time.perf_counter()
        try:
            session.run(None, {input_name: dummy_input})
        except Exception as e:
            result["status"] = "INFERENCE_FAILED"
            result["error"] = str(e)
            return result
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    
    lats = np.array(latencies_ms)
    
    result.update({
        "status": "COMPLETED",
        "mean_ms": round(float(np.mean(lats)), 2),
        "std_ms": round(float(np.std(lats)), 2),
        "min_ms": round(float(np.min(lats)), 2),
        "max_ms": round(float(np.max(lats)), 2),
        "p50_ms": round(float(np.percentile(lats, 50)), 2),
        "p90_ms": round(float(np.percentile(lats, 90)), 2),
        "p99_ms": round(float(np.percentile(lats, 99)), 2),
        "fps": round(1000.0 / float(np.mean(lats)), 1),
    })
    
    LOG.info("  Mean=%.2f ms | P50=%.2f ms | P99=%.2f ms | FPS=%.1f",
              result["mean_ms"], result["p50_ms"], result["p99_ms"], result["fps"])
    
    return result


def benchmark_e2e_ftr(video_path: Optional[Path], n_seconds: int = 30) -> Dict:
    """
    FTR pipeline uçtan uca benchmark.
    video_path yoksa sentetik video kullanır.
    """
    result = {"test": "e2e_ftr", "status": "NOT_STARTED"}
    
    try:
        import cv2
    except ImportError:
        result["status"] = "OPENCV_NOT_AVAILABLE"
        return result
    
    import cv2
    
    # Sentetik video veya gerçek video
    if video_path and video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        result["video"] = str(video_path)
    else:
        # Sentetik: sadece model latency test
        result["video"] = "synthetic_frames"
        
        # Detector modelini yükle (salt okunur)
        detector_path = PROJECT_ROOT / "models" / "detector.onnx"
        if not detector_path.is_file():
            result["status"] = "DETECTOR_NOT_FOUND"
            return result
        
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(detector_path), providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
        except Exception as e:
            result["status"] = "DETECTOR_LOAD_FAILED"
            result["error"] = str(e)
            return result
        
        frame_times_ms = []
        n_frames = min(n_seconds * 5, 50)  # 5 FPS sentetik
        
        for i in range(n_frames):
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            blob = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[None, ...]
            
            t0 = time.perf_counter()
            try:
                sess.run(None, {input_name: blob})
            except Exception:
                pass
            t1 = time.perf_counter()
            frame_times_ms.append((t1 - t0) * 1000)
        
        if frame_times_ms:
            result.update({
                "status": "COMPLETED",
                "n_frames": len(frame_times_ms),
                "mean_frame_ms": round(float(np.mean(frame_times_ms)), 2),
                "fps": round(1000.0 / float(np.mean(frame_times_ms)), 1),
            })
        else:
            result["status"] = "NO_FRAMES"
    
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Sinaptic5G — T4 CUDA Benchmark")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Tek model benchmark (opsiyonel)")
    parser.add_argument("--all-models", action="store_true",
                        help="Tüm modelleri benchmark yap")
    parser.add_argument("--n-runs", type=int, default=100,
                        help="Benchmark iterasyon sayısı")
    parser.add_argument("--n-warmup", type=int, default=20,
                        help="Warmup iterasyon sayısı")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch boyutu")
    parser.add_argument("--dry-run", action="store_true",
                        help="Hızlı dry-run (n_runs=5, n_warmup=2)")
    parser.add_argument("--fp16", action="store_true",
                        help="FP16 precision benchmark")
    parser.add_argument("--video", type=Path, default=None,
                        help="E2E benchmark için video dosyası")
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_t4_results.json"))
    
    args = parser.parse_args()
    
    if args.dry_run:
        args.n_runs = 5
        args.n_warmup = 2
    
    LOG.info("=== T4 CUDA Benchmark ===")
    
    # GPU bilgisi
    gpu_info = get_gpu_info()
    providers = get_available_providers()
    
    LOG.info("GPU: %s", gpu_info.get("name", "bulunamadı"))
    LOG.info("ONNX Providers: %s", providers)
    LOG.info("T4 mi?: %s", "✅" if gpu_info.get("is_t4") else "❌ (CPU veya farklı GPU)")
    
    all_results: Dict = {
        "gpu_info": gpu_info,
        "providers": providers,
        "benchmark_params": {
            "n_runs": args.n_runs,
            "n_warmup": args.n_warmup,
            "batch": args.batch,
            "fp16": args.fp16,
            "dry_run": args.dry_run,
        },
        "models": {},
        "t4_reference": T4_REFERENCE,
    }
    
    dtype = "float16" if args.fp16 else "float32"
    
    models_to_test: Dict[str, Dict] = {}
    
    if args.model_path:
        models_to_test["custom"] = {
            "path": str(args.model_path),
            "input_shape": [args.batch, 3, 640, 640],
            "dtype": dtype,
            "description": "Custom model",
        }
    
    if args.all_models or not args.model_path:
        for name, cfg in BENCHMARK_MODELS.items():
            models_to_test[name] = {
                **cfg,
                "input_shape": [args.batch] + cfg["input_shape"][1:],
                "dtype": dtype,
            }
    
    for model_name, cfg in models_to_test.items():
        model_path = PROJECT_ROOT / cfg["path"]
        LOG.info("\n--- Benchmark: %s ---", model_name)
        LOG.info("  Model: %s", model_path)
        LOG.info("  Input: %s, dtype=%s", cfg["input_shape"], cfg["dtype"])
        
        result = benchmark_model(
            model_path=model_path,
            input_shape=cfg["input_shape"],
            dtype=cfg["dtype"],
            n_runs=args.n_runs,
            n_warmup=args.n_warmup,
        )
        
        # Hedef latency karşılaştırma
        target_ms = cfg.get("target_latency_ms")
        if target_ms and result.get("mean_ms"):
            result["target_ms"] = target_ms
            result["target_met"] = result["mean_ms"] <= target_ms
            if not result["target_met"]:
                LOG.warning("  ⚠️  Hedef latency aşıldı: %.2f ms > %.2f ms",
                             result["mean_ms"], target_ms)
        
        all_results["models"][model_name] = result
    
    # E2E benchmark
    LOG.info("\n--- E2E FTR Pipeline Benchmark ---")
    e2e_result = benchmark_e2e_ftr(args.video)
    all_results["e2e_ftr"] = e2e_result
    
    # Özet
    LOG.info("\n=== Benchmark Özeti ===")
    for model_name, res in all_results["models"].items():
        if res.get("status") == "COMPLETED":
            target_ok = "✅" if res.get("target_met", True) else "❌"
            LOG.info("  %s [%s]: %.2f ms (P99=%.2f ms) | %.1f FPS %s",
                      model_name, "GPU" if res.get("using_gpu") else "CPU",
                      res["mean_ms"], res.get("p99_ms", 0), res["fps"], target_ok)
        else:
            LOG.info("  %s: %s", model_name, res.get("status"))
    
    # Rapor yaz
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("\nRapor: %s", args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
