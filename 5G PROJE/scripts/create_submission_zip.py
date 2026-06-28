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

"""
scripts/create_submission_zip.py
================================
Creates a clean code submission zip file excluding caches, virtual environments, 
and raw dataset folders while keeping YZ models, source codes, configs and evidence.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT.parent / "deliverables" / "SINAPTIC5G_CODE_SUBMISSION.zip"

EXCLUDE_DIRS = {
    ".venv-ftr",
    ".pytest_cache",
    "__pycache__",
    "tmp",
    "logs",
    "veriseti",
    "dataset",
    ".git",
    ".github"
}

EXCLUDE_FILES = {
    "download_errors.log",
}

def zip_folder():
    print(f"=== SİNAPTİC5G Code Submission Packager ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output ZIP: {OUTPUT_ZIP}")
    
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        file_count = 0
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Convert to Path
            root_path = Path(root)
            relative_root = root_path.relative_to(PROJECT_ROOT)
            
            # Check if any parent folder is in EXCLUDE_DIRS
            parts = relative_root.parts
            if any(part in EXCLUDE_DIRS for part in parts):
                continue
            
            # Special case for data folder: exclude raw, processed, keep splits
            if len(parts) > 0 and parts[0] == "data":
                # Only keep data/splits
                if not (len(parts) > 1 and parts[1] == "splits"):
                    continue
            
            for file in files:
                if file in EXCLUDE_FILES:
                    continue
                if file.endswith(".pyc") or file.endswith(".pyo") or file.endswith(".pyd"):
                    continue
                if file.startswith("."):
                    continue
                    
                file_path = root_path / file
                archive_name = relative_root / file
                
                # Check for other large files that are not models
                if file_path.stat().st_size > 10 * 1024 * 1024 and not file.endswith(".onnx") and not file.endswith(".tflite"):
                    print(f"[WARN] Skipping large file: {archive_name} ({file_path.stat().st_size / 1024 / 1024:.2f} MB)")
                    continue
                    
                zipf.write(file_path, archive_name)
                file_count += 1
                
        print(f"\n[OK] Successfully zipped {file_count} files into {OUTPUT_ZIP}")
        print(f"ZIP Size: {OUTPUT_ZIP.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    zip_folder()
