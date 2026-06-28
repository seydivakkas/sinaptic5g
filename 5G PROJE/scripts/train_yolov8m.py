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

"""F7 Task: Merge all preprocessed datasets and train YOLOv8m model."""

import os
import shutil
import random
import yaml
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

def clear_directory(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def copy_dataset_split(src_images: Path, src_labels: Path, dst_images: Path, dst_labels: Path, class_map: dict = None):
    if not src_images.is_dir() or not src_labels.is_dir():
        return
        
    for label_path in src_labels.glob("*.txt"):
        img_path = src_images / f"{label_path.stem}.jpg"
        if not img_path.is_file():
            img_path = src_images / f"{label_path.stem}.png"
            if not img_path.is_file():
                continue
                
        # Copy image
        shutil.copy2(img_path, dst_images / img_path.name)
        
        # Read and map labels if mapping is provided
        if class_map:
            mapped_lines = []
            content = label_path.read_text(encoding="utf-8", errors="replace")
            for line in content.splitlines():
                parts = line.split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id in class_map:
                    mapped_lines.append(f"{class_map[class_id]} {' '.join(parts[1:])}")
            with (dst_labels / label_path.name).open("w", encoding="utf-8") as f:
                f.write("\n".join(mapped_lines) + "\n")
        else:
            shutil.copy2(label_path, dst_labels / label_path.name)

def process_license_plates(real_plate_dir: Path, dst_root: Path, max_plates: int = 1000):
    print("Adding license plate images from turkish_plate_real dataset...")
    src_images = real_plate_dir / "images"
    src_labels = real_plate_dir / "label"
    
    if not src_labels.is_dir():
        print("Warning: turkish_plate_real labels not found.")
        return
        
    labels = sorted(list(src_labels.glob("*.txt")))
    random.seed(42)
    random.shuffle(labels)
    
    # Cap plates to prevent heavy imbalance
    labels = labels[:max_plates]
    
    n_total = len(labels)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    for idx, label_path in enumerate(labels):
        img_path = src_images / f"{label_path.stem}.jpg"
        if not img_path.is_file():
            img_path = src_images / f"{label_path.stem}.png"
            if not img_path.is_file():
                continue
                
        # Determine split
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"
            
        dst_images = dst_root / split / "images"
        dst_labels = dst_root / split / "labels"
        
        # Copy image with prefix
        new_img_name = f"plate_{img_path.name}"
        new_lbl_name = f"plate_{label_path.name}"
        
        shutil.copy2(img_path, dst_images / new_img_name)
        
        # Map class 0 (license_plate) to 8 (license_plate)
        content = label_path.read_text(encoding="utf-8", errors="replace")
        mapped_lines = []
        for line in content.splitlines():
            parts = line.split()
            if not parts:
                continue
            # Turkish plate real dataset has class 0, map to target class 8
            mapped_lines.append(f"8 {' '.join(parts[1:])}")
            
        with (dst_labels / new_lbl_name).open("w", encoding="utf-8") as f:
            f.write("\n".join(mapped_lines) + "\n")

def process_local_license_plates(local_labels_dir: Path, local_images_dir: Path, dst_root: Path):
    print("Processing license plates from local dataset...")
    if not local_labels_dir.is_dir():
        return
        
    for label_path in local_labels_dir.glob("*.txt"):
        img_path = local_images_dir / f"{label_path.stem}.jpg"
        if not img_path.is_file():
            img_path = local_images_dir / f"{label_path.stem}.png"
            if not img_path.is_file():
                continue
                
        content = label_path.read_text(encoding="utf-8", errors="replace")
        mapped_boxes = []
        for line in content.splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id == 0:  # license_plate in local dataset
                # Map to target class 8
                mapped_boxes.append(f"8 {' '.join(parts[1:])}")
                
        if mapped_boxes:
            new_img_name = f"local_plate_{img_path.name}"
            new_lbl_name = f"local_plate_{label_path.name}"
            
            # Since local dataset is train only, write to train split
            shutil.copy2(img_path, dst_root / "train" / "images" / new_img_name)
            with (dst_root / "train" / "labels" / new_lbl_name).open("w", encoding="utf-8") as f:
                f.write("\n".join(mapped_boxes) + "\n")

def check_merged_distribution(dst_root: Path):
    splits = ["train", "val", "test"]
    print("\n" + "=" * 60)
    print("  Merged Dataset Distribution")
    print("=" * 60)
    for split in splits:
        labels_dir = dst_root / split / "labels"
        class_counts = Counter()
        files = list(labels_dir.glob("*.txt"))
        for f in files:
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if parts:
                    class_counts[int(parts[0])] += 1
        print(f"Split {split:5s} | Images: {len(files):5d} | Classes: {dict(class_counts)}")
    print("=" * 60)

def main():
    merged_root = Path("data/processed/merged")
    
    # 1. Clear merged directory structure
    splits = ["train", "val", "test"]
    for split in splits:
        clear_directory(merged_root / split / "images")
        clear_directory(merged_root / split / "labels")
        
    # 2. Copy F1: Driver Actions
    print("Copying Driver Actions dataset...")
    for split in splits:
        copy_dataset_split(
            Path("data/processed/driver_actions") / split / "images",
            Path("data/processed/driver_actions") / split / "labels",
            merged_root / split / "images",
            merged_root / split / "labels"
        )
        
    # 3. Copy F2: Cigarette & Seatbelt
    print("Copying Cigarette & Seatbelt dataset...")
    for split in splits:
        copy_dataset_split(
            Path("data/processed/cigarette_seatbelt") / split / "images",
            Path("data/processed/cigarette_seatbelt") / split / "labels",
            merged_root / split / "images",
            merged_root / split / "labels"
        )
        
    # 4. Copy F5: Teknocan
    print("Copying Teknocan dataset...")
    for split in splits:
        copy_dataset_split(
            Path("data/processed/teknocan") / split / "images",
            Path("data/processed/teknocan") / split / "labels",
            merged_root / split / "images",
            merged_root / split / "labels"
        )
        
    # 5. Process local license plates
    process_local_license_plates(
        Path("dataset/labels/train"),
        Path("dataset/images/train"),
        merged_root
    )
    
    # 6. Copy turkish_plate_real dataset
    process_license_plates(
        Path("dataset/open_datasets/turkish_plate_real"),
        merged_root,
        max_plates=1200
    )
    
    # 7. Print distribution check
    check_merged_distribution(merged_root)
    
    # 8. Create data.yaml
    data_yaml = {
        "path": str(merged_root.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 9,
        "names": {
            0: "telefonla_konusma",
            1: "su_icme",
            2: "arkaya_bakma",
            3: "esneme",
            4: "sigara_icme",
            5: "emniyet_kemeri_ihlali",
            6: "teknocan",
            7: "bilgisayar",
            8: "license_plate"
        }
    }
    
    yaml_path = merged_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"Created dataset data.yaml at {yaml_path}")
    
    # 9. Train model
    print("\nStarting YOLOv8m training...")
    model = YOLO("yolov8m.pt")
    
    # Set models/runs/detector_v1 as output weights location
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=3,
        batch=16,
        imgsz=640,
        device=0,
        amp=True,
        freeze=10,
        project="models/runs",
        name="detector_v1"
    )
    print("Training finished successfully. Output weights saved under models/runs/detector_v1/weights/best.pt")

if __name__ == "__main__":
    main()
