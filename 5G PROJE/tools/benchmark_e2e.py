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
import asyncio
from pathlib import Path
import cv2
import numpy as np

# Ensure parent directories exist
Path("reports").mkdir(exist_ok=True)
Path("tests/smoke_input").mkdir(parents=True, exist_ok=True)
Path("tests/smoke_output").mkdir(parents=True, exist_ok=True)

# Add parent path to import ftr_main
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ftr_main import async_analyze_video

def create_synthetic_video(path, duration=4, fps=30, size=(640, 480)):
    print(f"[*] Creating synthetic video at {path} ({duration}s, {fps} fps)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    for _ in range(duration * fps):
        # 120 frames of black/neutral frame
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    print("[+] Synthetic video generated.")

def main():
    video_path = Path("tests/smoke_input/smoke_test.mp4")
    output_path = Path("tests/smoke_output/results.json")
    model_lock_path = Path("model_lock.json")
    
    # Generate video if missing
    if not video_path.is_file():
        create_synthetic_video(video_path)
        
    print("[*] Running end-to-end CPU performance benchmark (5 iterations)...")
    durations = []
    
    # Warmup
    try:
        asyncio.run(async_analyze_video(video_path, output_path, profile=False, enable_lstm=True))
    except Exception as e:
        print(f"[-] Warmup failed: {e}")
        sys.exit(1)
        
    for i in range(5):
        t0 = time.perf_counter()
        try:
            asyncio.run(async_analyze_video(video_path, output_path, profile=False, enable_lstm=True))
            duration = time.perf_counter() - t0
            durations.append(duration)
            print(f"    Run {i+1}/5: {duration:.2f}s")
        except Exception as e:
            print(f"[-] Run {i+1} failed: {e}")
            sys.exit(1)
            
    avg_duration = float(np.mean(durations))
    print(f"\n[+] Average E2E CPU latency: {avg_duration:.2f}s (Target: <= 8.0s)")
    
    # Save report
    report = {
        "benchmark_file": str(video_path),
        "iterations": 5,
        "runs_seconds": durations,
        "average_seconds": avg_duration,
        "target_seconds": 8.0,
        "is_compliant": avg_duration <= 8.0
    }
    
    report_path = Path("reports/e2e_latency_benchmark.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[+] Saved E2E latency benchmark to {report_path}")
    
    # Update model_lock.json status
    if avg_duration <= 8.0:
        print("[*] E2E CPU latency <= 8.0s threshold met. Updating model_lock.json...")
        if model_lock_path.is_file():
            lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
            lock["e2e_cpu_status"] = "ÖLÇÜLDÜ"
            model_lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
            print("[+] Successfully updated model_lock.json: e2e_cpu_status = ÖLÇÜLDÜ")
    else:
        print("[!] Warning: Average E2E CPU latency did not meet the <= 8.0s threshold.")

if __name__ == "__main__":
    main()
