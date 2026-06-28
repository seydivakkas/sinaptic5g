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
scripts/export_fp16_onnx.py — FP16 ONNX Quantization Scripti
=============================================================
Faz 4: Pipeline Optimizasyonu

UYARI: Üretim modeli models/detector.onnx (detector_v3) ASLA dokunulmaz.
FP16 export SADECE deneysel modeller için yapılır.

Desteklenen export hedefleri:
- detector_v4_full → models/runs/experiments/detector_v4_fp16.onnx
- crnn → models/runs/crnn_fp16.onnx
- cnn_lstm → models/runs/lstm_fp16.onnx

Kullanım:
    # Deneysel model için (güvenli):
    python scripts/export_fp16_onnx.py \
        --input models/runs/experiments/detector_v4_full_50ep/weights/best.onnx \
        --output models/runs/experiments/detector_v4_fp16.onnx \
        --model-type detector

    # Üretim modelini koruma testi:
    python scripts/export_fp16_onnx.py --self-test

Bağımlılıklar: onnx, onnxmltools (opsiyonel), onnxruntime
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

LOG = logging.getLogger("sinaptic5g.export_fp16")

# ─── Üretim Kilidi ────────────────────────────────────────────────────────────
# Bu dosyalar ASLA FP16 export hedefi olamaz
LOCKED_PRODUCTION_MODELS = {
    Path("models/detector.onnx").resolve(),
    Path("models/detector_v3.onnx").resolve(),
    Path("models/detector_v3_phase1.onnx").resolve(),
    Path("models/crnn.onnx").resolve(),
    Path("models/crnn_int8.onnx").resolve(),
    Path("models/cnn_lstm.onnx").resolve(),
    Path("models/lprnet.onnx").resolve(),
}


def _is_production_model(path: Path) -> bool:
    """Yol bir üretim modeline mi ait?"""
    return path.resolve() in LOCKED_PRODUCTION_MODELS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ─── FP16 Dönüştürücü ────────────────────────────────────────────────────────

def convert_to_fp16_onnxmltools(input_path: Path, output_path: Path) -> Dict:
    """onnxmltools ile FP16 quantization."""
    try:
        import onnx
        import onnxmltools
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        
        LOG.info("onnxmltools FP16 dönüştürme...")
        model = onnx.load(str(input_path))
        model_fp16 = convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, str(output_path))
        
        return {"method": "onnxmltools", "status": "SUCCESS"}
    except ImportError:
        LOG.warning("onnxmltools bulunamadı — fallback deneniyor")
        return {"method": "onnxmltools", "status": "NOT_AVAILABLE"}
    except Exception as e:
        return {"method": "onnxmltools", "status": "FAILED", "error": str(e)}


def convert_to_fp16_onnx_native(input_path: Path, output_path: Path) -> Dict:
    """onnx native FP16 dönüştürme (onnxmltools yoksa)."""
    try:
        import onnx
        from onnx import helper, TensorProto
        from onnx.numpy_helper import from_array, to_array
        
        LOG.info("ONNX native FP16 dönüştürme (initializer cast)...")
        model = onnx.load(str(input_path))
        
        # FP32 initializer'ları FP16'ya çevir
        converted = 0
        for initializer in model.graph.initializer:
            if initializer.data_type == TensorProto.FLOAT:
                arr = to_array(initializer).astype(np.float16)
                new_init = from_array(arr, name=initializer.name)
                initializer.CopyFrom(new_init)
                converted += 1
        
        onnx.save(model, str(output_path))
        return {"method": "onnx_native", "status": "SUCCESS", "converted_tensors": converted}
    except ImportError:
        return {"method": "onnx_native", "status": "NOT_AVAILABLE", "error": "onnx not installed"}
    except Exception as e:
        return {"method": "onnx_native", "status": "FAILED", "error": str(e)}


def verify_fp16_output(output_path: Path, input_path: Path) -> Dict:
    """FP16 modeli ONNX Runtime ile doğrular."""
    verify_result = {"verified": False}
    
    try:
        import onnxruntime as ort
        
        sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        input_info = sess.get_inputs()[0]
        input_name = input_info.name
        input_shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
        
        # FP16 input deneniyor
        dummy = np.zeros(input_shape, dtype=np.float32)
        try:
            output = sess.run(None, {input_name: dummy})
            verify_result["verified"] = True
            verify_result["output_shape"] = [list(o.shape) for o in output]
        except Exception as e:
            verify_result["error"] = str(e)
            verify_result["verified"] = False
        
        # Boyut karşılaştırma
        orig_size = input_path.stat().st_size
        fp16_size = output_path.stat().st_size
        compression_ratio = fp16_size / orig_size
        verify_result["original_size_mb"] = round(orig_size / 1024 / 1024, 2)
        verify_result["fp16_size_mb"] = round(fp16_size / 1024 / 1024, 2)
        verify_result["compression_ratio"] = round(compression_ratio, 3)
        
        LOG.info("  Orijinal boyut: %.1f MB", orig_size / 1024 / 1024)
        LOG.info("  FP16 boyut   : %.1f MB", fp16_size / 1024 / 1024)
        LOG.info("  Sıkıştırma   : %.2fx", 1.0 / compression_ratio if compression_ratio > 0 else 0)
        
    except Exception as e:
        verify_result["error"] = str(e)
    
    return verify_result


def self_test() -> int:
    """Üretim kilidi doğrulama testi."""
    LOG.info("=== Üretim Kilidi Self-Test ===")
    all_locked = True
    
    for locked_path in LOCKED_PRODUCTION_MODELS:
        if _is_production_model(locked_path):
            LOG.info("  ✅ KILITLI: %s", locked_path)
        else:
            LOG.error("  ❌ KİLİT HATASI: %s", locked_path)
            all_locked = False
    
    # Test: Kilitli modele yazmayı deneme
    test_path = Path("models/detector.onnx")
    if _is_production_model(test_path):
        LOG.info("✅ Üretim kilidi aktif — models/detector.onnx FP16 hedefi olamaz")
        return 0
    else:
        LOG.error("❌ Üretim kilidi BAŞARISIZ!")
        return 1


def export_fp16(
    input_path: Path,
    output_path: Path,
    verify: bool = True,
) -> Dict:
    """
    ONNX modelini FP16'ya dönüştürür.
    ÜRETİM MODELLERİNE ASLA YAZAMAZ.
    """
    results: Dict = {
        "input": str(input_path),
        "output": str(output_path),
        "production_lock_check": "PASSED",
        "status": "UNKNOWN",
    }
    
    # ─── Üretim Kilidi Kontrolü ──────────────────────────────────────────
    if _is_production_model(input_path):
        LOG.error("GÜVENLİK REDDİ: Girdi üretim modeli — FP16 dönüştürme reddedildi: %s", input_path)
        results["status"] = "REJECTED_PRODUCTION_INPUT"
        results["production_lock_check"] = "FAILED_INPUT"
        return results
    
    if _is_production_model(output_path):
        LOG.error("GÜVENLİK REDDİ: Çıktı üretim modeli üzerine yazılamaz: %s", output_path)
        results["status"] = "REJECTED_PRODUCTION_OUTPUT"
        results["production_lock_check"] = "FAILED_OUTPUT"
        return results
    
    if not input_path.is_file():
        LOG.error("Girdi modeli bulunamadı: %s", input_path)
        results["status"] = "INPUT_NOT_FOUND"
        return results
    
    # Çıktı dizinini oluştur
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # SHA256 girdi
    results["input_sha256"] = _sha256(input_path)
    results["input_size_mb"] = round(input_path.stat().st_size / 1024 / 1024, 2)
    
    LOG.info("FP16 dönüştürme: %s → %s", input_path.name, output_path.name)
    
    # onnxmltools önce dene, yoksa native
    conv_result = convert_to_fp16_onnxmltools(input_path, output_path)
    
    if conv_result["status"] not in ("SUCCESS",):
        conv_result = convert_to_fp16_onnx_native(input_path, output_path)
    
    results["conversion"] = conv_result
    
    if conv_result["status"] == "SUCCESS" and output_path.is_file():
        results["output_sha256"] = _sha256(output_path)
        
        if verify:
            verify_result = verify_fp16_output(output_path, input_path)
            results["verification"] = verify_result
            
            if verify_result.get("verified"):
                results["status"] = "COMPLETED"
                LOG.info("✅ FP16 export başarılı: %s", output_path)
            else:
                results["status"] = "VERIFICATION_FAILED"
                LOG.warning("⚠️  Doğrulama başarısız: %s", verify_result.get("error"))
        else:
            results["status"] = "COMPLETED_NO_VERIFY"
    else:
        results["status"] = "CONVERSION_FAILED"
        LOG.error("FP16 dönüştürme başarısız: %s", conv_result.get("error"))
    
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Sinaptic5G — FP16 ONNX Export")
    parser.add_argument("--input", type=Path, default=None,
                        help="Girdi ONNX model yolu (deneysel model)")
    parser.add_argument("--output", type=Path, default=None,
                        help="FP16 çıktı yolu (üretim dizinlerinden farklı olmalı)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Doğrulama adımını atla")
    parser.add_argument("--report-path", type=Path, default=Path("reports/fp16_export_report.json"))
    parser.add_argument("--self-test", action="store_true",
                        help="Üretim kilidi self-test çalıştır")
    
    args = parser.parse_args()
    
    if args.self_test:
        return self_test()
    
    if args.input is None or args.output is None:
        LOG.error("--input ve --output gereklidir (--self-test kullanmıyorsanız)")
        parser.print_help()
        return 1
    
    LOG.info("FP16 ONNX Export başlıyor...")
    LOG.info("  Üretim kilidi aktif: %d model kilitli", len(LOCKED_PRODUCTION_MODELS))
    
    results = export_fp16(
        input_path=args.input,
        output_path=args.output,
        verify=not args.no_verify,
    )
    
    # Rapor yaz
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Rapor: %s", args.report_path)
    LOG.info("Durum: %s", results.get("status"))
    
    return 0 if results.get("status") in ("COMPLETED", "COMPLETED_NO_VERIFY") else 1


if __name__ == "__main__":
    sys.exit(main())
