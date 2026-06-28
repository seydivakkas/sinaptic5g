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

"""Upload the local dataset to Roboflow.

This script uses the Roboflow Python SDK to create a project and upload the local
images and YOLO annotations.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from roboflow import Roboflow


def upload_dataset(api_key: str, workspace_name: str, project_name: str, dataset_dir: Path) -> None:
    print(f"Initializing Roboflow client...")
    rf = Roboflow(api_key=api_key)
    
    print(f"Loading workspace: {workspace_name}...")
    ws = rf.workspace(workspace_name)
    
    # Try to find or create the project
    project = None
    try:
        project = ws.project(project_name)
        print(f"Project '{project_name}' already exists.")
    except Exception:
        print(f"Project '{project_name}' not found. Creating a new one...")
        project = ws.create_project(
            project_name=project_name,
            project_type="object-detection",
            project_license="Public Domain",
            annotation="vehicles"
        )
        print(f"Created project: {project.id}")

    image_dir = dataset_dir / "images" / "train"
    label_dir = dataset_dir / "labels" / "train"

    if not image_dir.is_dir():
        print(f"Error: image directory not found at {image_dir}")
        return

    images = sorted(list(image_dir.glob("*.jpg")))
    print(f"Found {len(images)} images in {image_dir}.")

    uploaded_count = 0
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        annotation_path = str(label_path) if label_path.is_file() else None
        
        print(f"Uploading {image_path.name} (with annotation: {label_path.name if annotation_path else 'None'})...")
        try:
            project.upload(
                image_path=str(image_path),
                annotation_path=annotation_path,
                batch_name="Sinaptic5G_Upload",
                num_retry_uploads=3
            )
            uploaded_count += 1
        except Exception as e:
            print(f"Failed to upload {image_path.name}: {e}")

    print(f"Completed! Successfully uploaded {uploaded_count} of {len(images)} images.")


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    default_key = os.getenv("ROBOFLOW_API_KEY", "")

    parser = argparse.ArgumentParser(description="Upload dataset to Roboflow")
    parser.add_argument("--api-key", type=str, default=default_key, help="Roboflow private API key")
    parser.add_argument("--workspace", type=str, default="seydi-vakkas-eryilmaz", help="Roboflow workspace URL/slug")
    parser.add_argument("--project", type=str, default="sinaptic5g", help="Roboflow project URL/slug")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Path to dataset root folder")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Roboflow API key is not set. Please set ROBOFLOW_API_KEY in your .env file or pass it using --api-key.")
        return 1

    try:
        upload_dataset(
            api_key=args.api_key,
            workspace_name=args.workspace,
            project_name=args.project,
            dataset_dir=args.dataset
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
