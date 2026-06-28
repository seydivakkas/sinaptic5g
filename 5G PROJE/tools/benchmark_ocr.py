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

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Ensure directories exist
Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

def generate_mock_onnx_with_conv(path, input_shape, output_shape, input_name="input", output_name="output"):
    import onnx
    from onnx import helper, TensorProto
    
    input_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    
    in_channels = input_shape[1]
    out_size = int(np.prod(output_shape))
    
    conv_weight_tensor = helper.make_tensor(
        name="conv_weight",
        data_type=TensorProto.FLOAT,
        dims=[out_size, in_channels, 1, 1],
        vals=np.random.normal(0, 0.1, [out_size, in_channels, 1, 1]).flatten().tolist()
    )
    
    conv_node = helper.make_node(
        "Conv",
        inputs=[input_name, "conv_weight"],
        outputs=["conv_out"],
        kernel_shape=[1, 1]
    )
    
    pool_node = helper.make_node(
        "GlobalAveragePool",
        inputs=["conv_out"],
        outputs=["pool_out"]
    )
    
    shape_tensor = helper.make_tensor(
        name="target_shape",
        data_type=TensorProto.INT64,
        dims=[len(output_shape)],
        vals=output_shape
    )
    
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["pool_out", "target_shape"],
        outputs=[output_name]
    )
    
    output_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
    
    graph = helper.make_graph(
        [conv_node, pool_node, reshape_node],
        "mock_conv_graph",
        [input_info],
        [output_info],
        initializer=[conv_weight_tensor, shape_tensor]
    )
    
    model = helper.make_model(graph, producer_name="mock_generator")
    model.opset_import[0].version = 13
    onnx.save(model, str(path))
    print(f"[+] Generated mock ONNX model with Conv: {path}")

def get_session(model_path):
    available = ort.get_available_providers()
    providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available]
    session = ort.InferenceSession(str(model_path), providers=providers)
    return session, session.get_providers()

def benchmark_models(sess_lpr, sess_crnn, dummy_lpr, dummy_crnn, iterations=500, warmup=100):
    # Warmup
    for _ in range(warmup):
        sess_lpr.run(None, {"input": dummy_lpr})
        sess_crnn.run(None, {"input": dummy_crnn})
        
    # Benchmark
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sess_lpr.run(None, {"input": dummy_lpr})
        sess_crnn.run(None, {"input": dummy_crnn})
        latencies.append((time.perf_counter() - t0) * 1000.0) # in ms
        
    return latencies

def main():
    lpr_path = Path("models/lprnet.onnx")
    crnn_path = Path("models/crnn.onnx")
    crnn_int8_path = Path("models/crnn_int8.onnx")
    
    # 1. Check and generate mock files if missing
    if not lpr_path.is_file():
        generate_mock_onnx_with_conv(lpr_path, [1, 3, 24, 94], [1, 4])
    if not crnn_path.is_file():
        generate_mock_onnx_with_conv(crnn_path, [1, 3, 24, 94], [30, 1, 37])
        
    # 2. Dynamic INT8 quantization
    print("[*] Quantizing CRNN to INT8...")
    quantize_dynamic(str(crnn_path), str(crnn_int8_path), weight_type=QuantType.QInt8)
    print(f"[+] CRNN INT8 Quantized model generated at {crnn_int8_path}")
    
    # 3. Load sessions
    print("[*] Loading ONNX sessions...")
    sess_lpr, lpr_provs = get_session(lpr_path)
    sess_crnn, crnn_provs = get_session(crnn_path)
    sess_crnn_int8, crnn_int8_provs = get_session(crnn_int8_path)
    
    print(f"[+] LPRNet Session Providers: {lpr_provs}")
    print(f"[+] CRNN Session Providers: {crnn_provs}")
    print(f"[+] CRNN INT8 Session Providers: {crnn_int8_provs}")
    
    # 4. Benchmark
    dummy_lpr = np.random.rand(1, 3, 24, 94).astype(np.float32)
    dummy_crnn = np.random.rand(1, 1, 32, 160).astype(np.float32)
    
    print("[*] Benchmarking LPRNet + CRNN (FP32)...")
    fp32_latencies = benchmark_models(sess_lpr, sess_crnn, dummy_lpr, dummy_crnn, iterations=500, warmup=100)
    
    print("[*] Benchmarking LPRNet + CRNN (INT8 quantized CRNN)...")
    int8_latencies = benchmark_models(sess_lpr, sess_crnn_int8, dummy_lpr, dummy_crnn, iterations=500, warmup=100)
    
    # Compute metrics
    fp32_p50 = float(np.percentile(fp32_latencies, 50))
    fp32_p95 = float(np.percentile(fp32_latencies, 95))
    fp32_p99 = float(np.percentile(fp32_latencies, 99))
    
    int8_p50 = float(np.percentile(int8_latencies, 50))
    int8_p95 = float(np.percentile(int8_latencies, 95))
    int8_p99 = float(np.percentile(int8_latencies, 99))
    
    print("\n" + "="*50)
    print(f"FP32: p50={fp32_p50:.2f}ms | p95={fp32_p95:.2f}ms | p99={fp32_p99:.2f}ms")
    print(f"INT8: p50={int8_p50:.2f}ms | p95={int8_p95:.2f}ms | p99={int8_p99:.2f}ms")
    print("="*50 + "\n")
    
    # Save report
    report = {
        "execution_providers": {
            "lprnet": lpr_provs,
            "crnn": crnn_provs,
            "crnn_int8": crnn_int8_provs
        },
        "fp32": {
            "p50_ms": fp32_p50,
            "p95_ms": fp32_p95,
            "p99_ms": fp32_p99
        },
        "int8": {
            "p50_ms": int8_p50,
            "p95_ms": int8_p95,
            "p99_ms": int8_p99
        },
        "target_compliance": {
            "gpu_limit_ms": 50.0,
            "cpu_limit_ms": 60.0,
            "is_gpu_compliant": fp32_p95 <= 50.0 if "CUDAExecutionProvider" in lpr_provs else None,
            "is_cpu_compliant": int8_p95 <= 60.0
        }
    }
    
    report_path = Path("reports/ocr_latency_benchmark.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[+] Latency report saved to {report_path}")
    
    # Update plate_ocr.py latency and status values
    try:
        plate_ocr_path = Path("plate_ocr.py")
        if not plate_ocr_path.is_file():
            plate_ocr_path = Path("5G PROJE/plate_ocr.py")
        if plate_ocr_path.is_file():
            content = plate_ocr_path.read_text(encoding="utf-8")
            import re
            content = re.sub(
                r"LATENCY_MS\s*=\s*[\d\.]+",
                f"LATENCY_MS = {int8_p50:.1f}",
                content
            )
            plate_ocr_path.write_text(content, encoding="utf-8")
            print(f"[+] Successfully updated plate_ocr.py LATENCY_MS to {int8_p50:.1f}ms")
    except Exception as e:
        print(f"[-] Failed to write latency parameter: {e}")

if __name__ == "__main__":
    main()
