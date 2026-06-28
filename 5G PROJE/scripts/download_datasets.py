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
scripts/download_datasets.py
============================
Automates downloading Kaggle and Roboflow datasets declared in datasets.yaml.
Checks for dependencies, prompts for Roboflow API Key, and sets up folders.
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset definitions mapped from configs/datasets.yaml
ROBOFLOW_DATASETS = {
    "driver_distraction_v3": {
        "workspace": "areeba-fmpau",
        "project": "driver-distraction-detection-jsu2o",
        "version": 3,
        "root": "dataset/open_datasets/driver_distraction/datasets"
    },
    "driver_drowsiness_v4": {
        "workspace": "driver-drowsiness-59y8h",
        "project": "driver_drowsiness_yolo-2duii",
        "version": 4,
        "root": "dataset/open_datasets/driver_drowsiness"
    },
    "cigarette_smokers_v5": {
        "workspace": "smoking-t7kym",
        "project": "cigarette-smokers-cnbaf",
        "version": 5,
        "root": "data/raw/cigarette"
    },
    "detect_seatbelt_v7": {
        "workspace": "seatbelt-y0rvu",
        "project": "detect-seatbelt-whvbz",
        "version": 7,
        "root": "data/raw/seatbelt"
    },
    "seatbelt_v1": {
        "workspace": "seatbelt-y0rvu",
        "project": "seat-belt-detection-c3csr-6gbyf",
        "version": 1,
        "root": "data/raw/seatbelt_v1"
    }
}

KAGGLE_DATASETS = {
    "kaggle_turkish_license_plate_2776891": {
        "slug": "smaildurcan/turkish-license-plate-dataset",
        "root": "dataset/open_datasets/turkish_plate_real"
    }
}


def install_package(pkg_name: str):
    print(f"Installing missing dependency: {pkg_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
    except Exception as e:
        print(f"Failed to install {pkg_name} automatically: {e}")
        print(f"Please install it manually: pip install {pkg_name}")
        sys.exit(1)


def check_dependencies():
    try:
        import roboflow
    except ImportError:
        install_package("roboflow")

    try:
        import kaggle
    except ImportError:
        install_package("kaggle")


def download_roboflow(api_key: str):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)

    for name, info in ROBOFLOW_DATASETS.items():
        dest_dir = PROJECT_ROOT / info["root"]
        if dest_dir.exists() and any(dest_dir.iterdir()):
            print(f"[SKIP] Roboflow dataset '{name}' is already downloaded at {info['root']}")
            continue

        print(f"\n[START] Downloading Roboflow dataset: {name} (Version {info['version']})...")
        try:
            # Recreate target directory
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            project = rf.workspace(info["workspace"]).project(info["project"])
            dataset = project.version(info["version"]).download("yolov8", location=str(dest_dir))
            
            print(f"[OK] Successfully downloaded '{name}' to {info['root']}")
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")


def download_kaggle():
    # Make sure kaggle credentials are set up
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.is_file():
        print("\n[WARNING] Kaggle credentials (kaggle.json) not found in ~/.kaggle/")
        print("Please follow these steps:")
        print("  1. Go to https://www.kaggle.com/ -> Your Account -> Create New API Token")
        print(f"  2. Save the downloaded 'kaggle.json' to '{kaggle_json}'")
        print("  3. Re-run this script to download Kaggle datasets.")
        return

    for name, info in KAGGLE_DATASETS.items():
        dest_dir = PROJECT_ROOT / info["root"]
        if dest_dir.exists() and any(dest_dir.iterdir()):
            print(f"[SKIP] Kaggle dataset '{name}' is already downloaded at {info['root']}")
            continue

        print(f"\n[START] Downloading Kaggle dataset: {info['slug']}...")
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Use kaggle CLI to download
            zip_path = dest_dir / "dataset.zip"
            cmd = ["kaggle", "datasets", "download", "-d", info["slug"], "-p", str(dest_dir)]
            print(f"Running command: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            
            # Find the downloaded zip file (it might have a custom name)
            zip_files = list(dest_dir.glob("*.zip"))
            if not zip_files:
                print(f"[ERROR] Zip file not found after downloading {info['slug']}")
                continue
                
            downloaded_zip = zip_files[0]
            print(f"Extracting {downloaded_zip.name} to {info['root']}...")
            with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
                
            # Clean up the zip file
            downloaded_zip.unlink()
            print(f"[OK] Successfully downloaded and extracted '{name}' to {info['root']}")
        except Exception as e:
            print(f"[ERROR] Failed to download Kaggle dataset {name}: {e}")


def main():
    print("=== SİNAPTİC5G Dataset Downloader ===")
    check_dependencies()

    # Get Roboflow API key
    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key:
        api_key = input("Please enter your Roboflow Private API Key: ").strip()

    if not api_key:
        print("[ERROR] Roboflow API Key is required. Exiting.")
        return 1

    # Download datasets
    download_roboflow(api_key)
    download_kaggle()
    
    print("\n[COMPLETE] Downloader process finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
