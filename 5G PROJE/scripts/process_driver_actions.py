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

"""F1 Task: Process Driver Distraction/Drowsiness Dataset.

Maps YOLO labels to FTR requirements:
  2 (phone)         -> 0: telefonla_konusma
  4 (drinking)      -> 1: su_icme
  5 (reaching)      -> 2: arkaya_bakma
  8, 9 (closed/yawn)-> 3: esneme
Filters unmapped images. Augments minority classes to reach min 200 samples.
"""

from __future__ import annotations

import os
import shutil
import random
from collections import Counter
from pathlib import Path
import cv2
import numpy as np


CLASS_MAPPING = {
    2: 0,  # phone -> telefonla_konusma
    4: 1,  # drinking -> su_icme
    5: 2,  # reaching behind -> arkaya_bakma
    8: 3,  # eyes closed -> esneme
    9: 3,  # yawning -> esneme
}


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    row, col, ch = image.shape
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch) * 255
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def augment_image(image: np.ndarray, aug_type: int) -> tuple[np.ndarray, bool]:
    """Apply simple image augmentations. Returns (augmented_image, requires_bbox_flip)."""
    if aug_type == 1:
        # 1. Horizontal Flip (requires x_center = 1.0 - x_center)
        return cv2.flip(image, 1), True
    elif aug_type == 2:
        # 2. Brightness Increase
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v.astype(np.int32) + 30, 0, 255).astype(np.uint8)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR), False
    elif aug_type == 3:
        # 3. Brightness Decrease
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v.astype(np.int32) - 30, 0, 255).astype(np.uint8)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR), False
    elif aug_type == 4:
        # 4. Contrast Increase
        alpha = 1.2
        return np.clip(alpha * image, 0, 255).astype(np.uint8), False
    else:
        # 5. Gaussian Noise
        return add_gaussian_noise(image), False


def process_split(src_dir: Path, dst_dir: Path, is_train: bool) -> dict:
    src_images = src_dir / "images"
    src_labels = src_dir / "labels"
    
    dst_images = dst_dir / "images"
    dst_labels = dst_dir / "labels"
    
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    class_counts = Counter()

    if not src_labels.is_dir():
        print(f"No labels directory found at {src_labels}")
        return {"processed": 0, "counts": class_counts}

    labels = sorted(list(src_labels.glob("*.txt")))
    print(f"Scanning {len(labels)} label files in {src_labels}...")

    for label_path in labels:
        image_path = src_images / f"{label_path.stem}.jpg"
        if not image_path.is_file():
            # Try png fallback
            image_path = src_images / f"{label_path.stem}.png"
            if not image_path.is_file():
                continue
                
        # Read annotation and map classes
        mapped_boxes = []
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id in CLASS_MAPPING:
                target_class = CLASS_MAPPING[class_id]
                mapped_boxes.append((target_class, parts[1], parts[2], parts[3], parts[4]))
                class_counts[target_class] += 1

        # Save if contains at least one mapped class
        if mapped_boxes:
            shutil.copy2(image_path, dst_images / image_path.name)
            new_label_file = dst_labels / label_path.name
            with new_label_file.open("w", encoding="utf-8") as f:
                for box in mapped_boxes:
                    f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")
            processed_files.append((image_path.name, label_path.name, mapped_boxes))

    # For train set: apply augmentations to minority classes
    if is_train:
        print("\nChecking class distribution for training split...")
        print(f"Original mapped counts: {dict(class_counts)}")
        
        # Identify classes with < 200 samples
        minority_classes = [c for c, count in class_counts.items() if count < 200]
        if minority_classes:
            print(f"Minority classes requiring augmentation (count < 200): {minority_classes}")
            
            # Group files by classes they contain
            files_to_augment = []
            for img_name, lbl_name, boxes in processed_files:
                contains_minority = any(box[0] in minority_classes for box in boxes)
                if contains_minority:
                    files_to_augment.append((img_name, lbl_name, boxes))
                    
            print(f"Applying 5x augmentation to {len(files_to_augment)} images containing minority classes...")
            
            for img_name, lbl_name, boxes in files_to_augment:
                image_file = dst_images / img_name
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                # Perform 5 augmentations
                for aug_idx in range(1, 6):
                    aug_img, flip_coords = augment_image(image, aug_idx)
                    aug_img_name = f"{Path(img_name).stem}_aug{aug_idx}.jpg"
                    aug_lbl_name = f"{Path(lbl_name).stem}_aug{aug_idx}.txt"
                    
                    # Save augmented image
                    cv2.imwrite(str(dst_images / aug_img_name), aug_img)
                    
                    # Update bounding boxes and save augmented label
                    with (dst_labels / aug_lbl_name).open("w", encoding="utf-8") as f:
                        for box in boxes:
                            class_id = box[0]
                            # Bbox parameters
                            x_center = float(box[1])
                            y_center = float(box[2])
                            width = float(box[3])
                            height = float(box[4])
                            
                            if flip_coords:
                                x_center = 1.0 - x_center
                                
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            # Count the augmented instances
                            class_counts[class_id] += 1

    print(f"Split completed. Processed images: {len(processed_files)}")
    print(f"Final mapped counts (including augmentations): {dict(class_counts)}")
    return {"processed": len(processed_files), "counts": class_counts}


def main() -> int:
    src_root = Path("dataset/open_datasets/driver_distraction/datasets")
    dst_root = Path("data/processed/driver_actions")

    # Map dataset/open_datasets/driver_distraction/datasets/valid -> data/processed/driver_actions/val
    splits = [
        ("train", "train", True),
        ("valid", "val", False),
        ("test", "test", False),
    ]

    total_processed = 0
    all_counts = Counter()

    for src_name, dst_name, is_train in splits:
        print(f"\nProcessing split: {src_name} -> {dst_name}")
        result = process_split(src_root / src_name, dst_root / dst_name, is_train)
        total_processed += result["processed"]
        all_counts.update(result["counts"])

    print("\n" + "=" * 60)
    print("  Dataset Processing Summary (F1 Task)")
    print("=" * 60)
    print(f"Total original processed images: {total_processed}")
    print(f"Total mapped classes: {dict(all_counts)}")
    print("Output directory: data/processed/driver_actions/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
