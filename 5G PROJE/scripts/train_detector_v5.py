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
scripts/train_detector_v5.py — GPU-Accelerated Training Script for detector_v5
=============================================================================
Phase 3 & 4: Model Training and GPU Optimization

This script:
1. Verifies CUDA availability.
2. Records system hardware and software versions to environment.txt.
3. Loads configs/train_detector_v5.yaml parameters.
4. Registers custom callbacks to update reports/detector_v5_training_progress.json on each epoch.
5. Runs 60 epochs of YOLOv8m training (or 1 epoch on CPU/GPU for --smoke-run).
6. Writes training execution logs to reports/detector_v5_training_log.md.

Usage:
    python scripts/train_detector_v5.py [--smoke-run] [--config configs/train_detector_v5.yaml]
"""

import os
import sys
import shutil
import json
import yaml
import argparse
from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]

def record_environment(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    env_txt = run_dir / "environment.txt"
    
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None"
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if cuda_avail else 0.0
    
    try:
        import onnxruntime
        ort_ver = onnxruntime.__version__
    except ImportError:
        ort_ver = "Not Installed"
        
    lines = [
        f"Python Version: {sys.version}",
        f"PyTorch Version: {torch.__version__}",
        f"Ultralytics Version: {ultralytics.__version__}",
        f"ONNX Runtime Version: {ort_ver}",
        f"CUDA Available: {cuda_avail}",
        f"CUDA Version: {torch.version.cuda if cuda_avail else 'N/A'}",
        f"GPU Name: {gpu_name}",
        f"GPU VRAM: {vram:.2f} GB",
    ]
    
    with open(env_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Recorded environment info to {env_txt}")
    return lines

def main():
    parser = argparse.ArgumentParser(description="detector_v5 Training Script")
    parser.add_argument("--smoke-run", action="store_true", help="Run 1 epoch as dry-run to verify training works")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/train_detector_v5.yaml", help="Path to config YAML")
    args_cli = parser.parse_args()
    
    if not args_cli.config.is_file():
        print(f"ERROR: Configuration file not found at {args_cli.config}")
        return 1
        
    # Load configuration parameters
    with open(args_cli.config, "r", encoding="utf-8") as f:
        args_train = yaml.safe_load(f)
        
    run_dir = PROJECT / "models/runs/experiments/detector_v5_60ep"
    if args_cli.smoke_run:
        run_dir = PROJECT / "models/runs/experiments/detector_v5_smoke"
        
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Record environment metadata
    env_lines = record_environment(run_dir)
    shutil.copy2(args_cli.config, run_dir / "run_recipe.yaml")
    
    # Clear old training progress
    progress_file = PROJECT / "reports/detector_v5_training_progress.json"
    if progress_file.is_file():
        try:
            progress_file.unlink()
        except Exception:
            pass
            
    # Write pre-training log entry
    train_log_md = PROJECT / "reports/detector_v5_training_log.md"
    train_log_md.parent.mkdir(parents=True, exist_ok=True)
    with open(train_log_md, "w", encoding="utf-8") as f:
        f.write("# SİNAPTİC5G — detector_v5 Training Log\n\n")
        f.write(f"> **Model:** YOLOv8m (Candidate v5)\n")
        f.write(f"> **Mode:** {'Smoke-Run' if args_cli.smoke_run else 'Full-Training (60 Epochs)'}\n\n")
        f.write("## 1. Eğitim Ortamı (Environment)\n\n")
        for line in env_lines:
            f.write(f"* {line}\n")
        f.write("\n## 2. Eğitim Konfigürasyonu (Recipe)\n\n")
        f.write("```yaml\n")
        with open(args_cli.config, "r", encoding="utf-8") as rf:
            f.write(rf.read())
        f.write("```\n\n")
        f.write("## 3. Eğitim İlerleme Kaydı (Training Progress)\n\n")
        f.write("Eğitim başlatıldı...\n")
        
    # Prepare custom training callback to record epoch progress
    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        
        # Loss metrics
        loss_dict = {}
        try:
            if hasattr(trainer, 'loss_items'):
                li = trainer.loss_items
                if hasattr(li, 'items'):
                    loss_dict = {k: float(v) for k, v in li.items()}
                elif hasattr(li, 'tolist'):
                    # if it's a tensor/array, name them by index
                    lst = li.tolist()
                    if isinstance(lst, list):
                        loss_dict = {f"loss_{i}": float(v) for i, v in enumerate(lst)}
                    else:
                        loss_dict = {"loss": float(lst)}
                else:
                    loss_dict = {"loss": float(li)}
            elif hasattr(trainer, 'tloss'):
                tl = trainer.tloss
                if hasattr(tl, 'tolist'):
                    lst = tl.tolist()
                    if isinstance(lst, list):
                        loss_dict = {f"loss_{i}": float(v) for i, v in enumerate(lst)}
                    else:
                        loss_dict = {"loss": float(lst)}
                else:
                    loss_dict = {"loss": float(tl)}
        except Exception as e:
            print(f"Warning: Failed to extract loss metrics: {e}")
            
        # Validation metrics
        metrics_dict = {}
        try:
            if hasattr(trainer, 'metrics') and trainer.metrics:
                if hasattr(trainer.metrics, 'items'):
                    for k, v in trainer.metrics.items():
                        metrics_dict[str(k)] = float(v)
                elif hasattr(trainer.metrics, 'tolist'):
                    metrics_dict["metrics"] = trainer.metrics.tolist()
        except Exception as e:
            print(f"Warning: Failed to extract validation metrics: {e}")
                
        progress_entry = {
            "epoch": epoch,
            "losses": loss_dict,
            "metrics": metrics_dict
        }
        
        # Read, append and rewrite progress
        prog_data = []
        if progress_file.is_file():
            try:
                prog_data = json.loads(progress_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        prog_data.append(progress_entry)
        
        try:
            progress_file.write_text(json.dumps(prog_data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"Warning: Failed to write progress file: {e}")
            
    # Setup YOLO model
    print("\nInitializing YOLOv8m model...")
    model = YOLO("yolov8m.pt")
    
    # Register callback
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Adjust arguments for smoke-run if needed
    if args_cli.smoke_run:
        print("\nRunning in smoke-run mode (1 epoch, batch=2, fraction=0.01, CPU or GPU)...")
        args_train["epochs"] = 1
        args_train["batch"] = 2
        args_train["imgsz"] = 320
        args_train["device"] = 0 if torch.cuda.is_available() else "cpu"
        args_train["name"] = "detector_v5_smoke"
        args_train["fraction"] = 0.01
        
    print("\nStarting YOLOv8m training for detector_v5...")
    try:
        results = model.train(**args_train)
        print("\nTraining completed successfully!")
        
        with open(train_log_md, "a", encoding="utf-8") as f:
            f.write(f"\nEğitim başarıyla tamamlandı. Çıktılar `{args_train['project']}/{args_train['name']}/` dizinine kaydedildi.\n")
            
        # If full run, link best.pt to candidates folder
        if not args_cli.smoke_run:
            candidate_dir = PROJECT / "models/candidates/detector_v5"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            best_weights = run_dir / "weights/best.pt"
            if best_weights.is_file():
                shutil.copy2(best_weights, candidate_dir / "best.pt")
                print(f"Copied best weights to {candidate_dir / 'best.pt'}")
                
        return 0
    except Exception as e:
        print(f"\nERROR: Training failed with exception: {e}")
        with open(train_log_md, "a", encoding="utf-8") as f:
            f.write(f"\nEğitim hatayla durdu: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
