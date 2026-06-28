# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
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

import asyncio
import time
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT))

from ftr_main import async_analyze_video, verify_model_lock

async def run_benchmark():
    input_file = PROJECT / "tests/smoke_input/smoke_test.mp4"
    output_file = PROJECT / "tests/smoke_output/results.json"
    
    if not input_file.is_file():
        print(f"Error: input file {input_file} is absent.")
        return 1
        
    print("Verifying model lock before benchmarking...")
    try:
        verify_model_lock()
        print("Model lock verification successful.")
    except Exception as e:
        print(f"Model lock verification failed: {e}")
        return 1
        
    iterations = 5
    runs = []
    
    print(f"Starting E2E Latency Benchmark ({iterations} iterations)...")
    
    # Warmup
    print("Warmup iteration...")
    await async_analyze_video(input_file, output_file, profile=False, enable_lstm=False)
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")
        t_start = time.perf_counter()
        await async_analyze_video(input_file, output_file, profile=False, enable_lstm=False)
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        runs.append(elapsed)
        print(f"Iteration {i+1} completed in {elapsed:.4f} seconds.")
        
    avg_seconds = sum(runs) / len(runs)
    target_seconds = 8.0
    is_compliant = avg_seconds <= target_seconds
    
    payload = {
        "benchmark_file": "tests/smoke_input/smoke_test.mp4",
        "iterations": iterations,
        "runs_seconds": runs,
        "average_seconds": avg_seconds,
        "target_seconds": target_seconds,
        "is_compliant": is_compliant
    }
    
    json_path = PROJECT / "reports/detector_v4_e2e_latency_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON benchmark to {json_path}")
    
    md_lines = [
        "# Offline FTR E2E Latency Benchmark (detector_v4)",
        f"\n* **Tarih:** 2026-06-21",
        f"* **Test Dosyası:** `tests/smoke_input/smoke_test.mp4`",
        f"* **Yineleme Sayısı:** {iterations}",
        f"\n## Sonuçlar",
        f"| Yineleme | Süre (Saniye) |",
        f"|---|---|",
    ]
    for idx, run_time in enumerate(runs):
        md_lines.append(f"| {idx+1} | {run_time:.4f}s |")
        
    md_lines.extend([
        f"\n## Özet",
        f"* **Ortalama İşleme Süresi:** {avg_seconds:.4f}s",
        f"* **Hedef Süre:** {target_seconds:.4f}s",
        f"* **Kriter Uyumluluğu:** {'TAMAMLANDI' if is_compliant else 'UYARI (8 saniye hedefi CPU üzerinde aşıldı)'}",
        f"\n## Değerlendirme ve Notlar",
    ])
    
    if is_compliant:
        md_lines.append("Sistem 8 saniyelik CPU hedef süre sınırını başarıyla karşılamaktadır.")
    else:
        md_lines.append("Sistem CPU üzerinde 8 saniyelik hedefi karşılamamaktadır. Ancak, ftr_main.py içerisinde bulunan dinamik stride mekanizması (Adaptive Stride) sayesinde GPU/CUDA Execution Provider aktifleştiğinde başarımın 8 saniyenin altına inmesi hedeflenmektedir (`HEDEF`).")
        
    md_path = PROJECT / "reports/detector_v4_e2e_latency_benchmark.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved Markdown report to {md_path}")
    
    return 0

if __name__ == "__main__":
    asyncio.run(run_benchmark())
