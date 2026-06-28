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
scripts/benchmark_e2e_v5.py — GPU + CPU E2E Gecikme Benchmarki (detector_v5)
=============================================================================
Hem GPU (CUDA) hem de CPU modlarinda sirayla benchmark calistirir.
Sonuclari reports/ altina JSON + Markdown olarak yazar.

Kullanim:
    python scripts/benchmark_e2e_v5.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

# ONNX Runtime provider mocking helper
import onnxruntime as ort
_original_get_available_providers = ort.get_available_providers

from ftr_main import async_analyze_video, verify_model_lock


async def run_single_benchmark(input_file: Path, output_file: Path) -> float:
    """Tek bir benchmark iterasyonu calistirir, sure doner."""
    t_start = time.perf_counter()
    await async_analyze_video(input_file, output_file, profile=False, enable_lstm=False)
    return time.perf_counter() - t_start


async def run_suite(mode: str, iterations: int, input_file: Path, output_file: Path) -> list[float]:
    """Belirli bir modda warmup + N iterasyon calistirir."""
    print(f"\n>>> [{mode} Modu] Baslatiliyor...")
    
    # Mock/Restore providers
    if mode == "CPU":
        ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    else:
        ort.get_available_providers = _original_get_available_providers
        
    # Warmup
    print("Warmup iterasyonu...")
    await run_single_benchmark(input_file, output_file)
    
    runs = []
    for i in range(iterations):
        elapsed = await run_single_benchmark(input_file, output_file)
        runs.append(elapsed)
        print(f"  Iterasyon {i+1}/{iterations}: {elapsed:.3f}s")
        
    return runs


async def main():
    input_file = PROJECT / "tests" / "smoke_input" / "smoke_test.mp4"
    output_file = PROJECT / "tests" / "smoke_output" / "results_benchmark.json"
    report_json = PROJECT / "reports" / "detector_v5_e2e_latency_benchmark.json"
    report_md = PROJECT / "reports" / "detector_v5_e2e_latency_benchmark.md"

    if not input_file.is_file():
        print(f"HATA: Girdi dosyasi bulunamadi: {input_file}")
        return 1

    # Model kilit dogrulama
    print("Model kilidi dogrulaniyor...")
    try:
        verify_model_lock()
        print("Model kilidi dogrulandi.")
    except Exception as e:
        print(f"Model kilit hatasi: {e}")
        return 1

    # Cihaz ve GPU Bilgisi
    available_providers = _original_get_available_providers()
    has_cuda = "CUDAExecutionProvider" in available_providers
    gpu_name = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    iterations = 5
    target_seconds = 8.0

    print(f"Sistem Bilgisi | CUDA Destegi: {has_cuda} | GPU: {gpu_name}")
    print(f"Benchmark plani: GPU (varsa) ve CPU modlarinda {iterations}'er iterasyon.")

    # 1. GPU (CUDA) Benchmark'ı
    gpu_runs = []
    if has_cuda:
        gpu_runs = await run_suite("GPU (CUDA)", iterations, input_file, output_file)
    else:
        print("\n[GPU (CUDA) Modu] CUDA saglayicisi bulunmadigi icin GPU benchmark atlandi.")

    # 2. CPU Benchmark'ı
    cpu_runs = await run_suite("CPU", iterations, input_file, output_file)

    # Orijinal fonksiyonu geri yukle
    ort.get_available_providers = _original_get_available_providers

    # Hesaplamalar
    results = {}
    
    if gpu_runs:
        gpu_avg = sum(gpu_runs) / len(gpu_runs)
        gpu_min = min(gpu_runs)
        gpu_max = max(gpu_runs)
        gpu_compliant = gpu_avg <= target_seconds
        results["gpu"] = {
            "runs": [round(r, 4) for r in gpu_runs],
            "average": round(gpu_avg, 4),
            "min": round(gpu_min, 4),
            "max": round(gpu_max, 4),
            "compliant": gpu_compliant
        }
    else:
        results["gpu"] = None

    cpu_avg = sum(cpu_runs) / len(cpu_runs)
    cpu_min = min(cpu_runs)
    cpu_max = max(cpu_runs)
    cpu_compliant = cpu_avg <= target_seconds
    results["cpu"] = {
        "runs": [round(r, 4) for r in cpu_runs],
        "average": round(cpu_avg, 4),
        "min": round(cpu_min, 4),
        "max": round(cpu_max, 4),
        "compliant": cpu_compliant
    }

    # JSON Payload
    payload = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": "detector_v5",
        "gpu_name": gpu_name if has_cuda else "N/A",
        "cuda_available": has_cuda,
        "benchmark_file": str(input_file.relative_to(PROJECT)),
        "iterations": iterations,
        "target_seconds": target_seconds,
        "results": results
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON rapor kaydedildi: {report_json}")

    # Markdown Raporu
    md = f"""# SiNAPTiC5G - detector_v5 E2E Gecikme Benchmark Raporu

> **Tarih:** {payload['date']}
> **Model:** detector_v5 (YOLOv8m ONNX)
> **GPU:** {gpu_name if has_cuda else 'N/A'}
> **CUDA Destegi:** {'Mevcut' if has_cuda else 'Yok'}

## 1. Mod Bazli Performans Tablosu

| Mod | Ortalama Sure | En Hizli | En Yavas | Hedef | Uyumluluk |
|---|:---:|:---:|:---:|:---:|:---:|
"""
    if results["gpu"]:
        gpu_status = "GEÇTİ ✅" if results["gpu"]["compliant"] else "KALDI ❌"
        md += f"| **GPU (CUDA)** | **{results['gpu']['average']:.4f}s** | {results['gpu']['min']:.4f}s | {results['gpu']['max']:.4f}s | {target_seconds:.1f}s | {gpu_status} |\n"
    else:
        md += f"| **GPU (CUDA)** | *N/A (CUDA Yok)* | - | - | {target_seconds:.1f}s | - |\n"
        
    cpu_status = "GEÇTİ ✅" if results["cpu"]["compliant"] else "KALDI ❌"
    md += f"| **CPU** | **{results['cpu']['average']:.4f}s** | {results['cpu']['min']:.4f}s | {results['cpu']['max']:.4f}s | {target_seconds:.1f}s | {cpu_status} |\n"

    md += """
## 2. Iterasyon Detaylari

| Iterasyon | GPU (CUDA) Sure | CPU Sure |
|:---------:|:---------------:|:--------:|
"""
    for i in range(iterations):
        gpu_val = f"{gpu_runs[i]:.4f}s" if gpu_runs else "-"
        cpu_val = f"{cpu_runs[i]:.4f}s"
        md += f"| {i+1} | {gpu_val} | {cpu_val} |\n"

    md += f"""
## 3. Degerlendirme ve Analiz

1. **GPU (CUDA) Performansi:**
"""
    if results["gpu"]:
        if results["gpu"]["compliant"]:
            md += f"   - GPU modunda ortalama **{results['gpu']['average']:.2f}s** E2E sure ile 8.0s hedefini basariyla karsilamaktadir. ✅\n"
        else:
            md += f"   - GPU modunda ortalama **{results['gpu']['average']:.2f}s** ile hedef asilmistir. ❌\n"
    else:
        md += "   - Sistemde CUDA uyumlu GPU bulunamadigi veya etkinlestirilemedigi icin GPU testi yapilamamistir.\n"

    md += f"""
2. **CPU Performansi:**
   - CPU modunda ortalama **{results['cpu']['average']:.2f}s** E2E sure olculmustur.
"""
    if results["cpu"]["compliant"]:
        md += f"   - CPU'da uyarlanabilir kare tarama (adaptive stride = 1.5s) ve azaltilmis warmup sayesinde **{target_seconds:.1f}s** hedefini basariyla karsilamaktadir. ✅\n"
    else:
        md += f"   - CPU'da ortalama **{results['cpu']['average']:.2f}s** surdugunden {target_seconds:.1f}s hedefini asmaktadir. Canli ortamda GPU/CUDA kullanimi zorunludur. ❌\n"

    md += f"""
---
OZEL LiSANS - TUM HAKLAR SAKLIDIR
Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
"""

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown rapor kaydedildi: {report_md}")

    # Ekrana Ozet Bas
    print(f"\n{'='*60}")
    print("  detector_v5 E2E GPU vs CPU Latency Summary")
    print('='*60)
    if results['gpu']:
        print(f"GPU (CUDA) Ortalama : {results['gpu']['average']:.3f}s (Hedef: {target_seconds:.1f}s) -> {'GEÇTİ' if results['gpu']['compliant'] else 'GEÇMEDİ'}")
    print(f"CPU Ortalama        : {results['cpu']['average']:.3f}s (Hedef: {target_seconds:.1f}s) -> {'GEÇTİ' if results['cpu']['compliant'] else 'GEÇMEDİ'}")
    print('='*60)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
