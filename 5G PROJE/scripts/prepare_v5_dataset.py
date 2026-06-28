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
scripts/prepare_v5_dataset.py — detector_v5 Dataset Preparation & Class Balancing
================================================================================
Phase 2: Dataset Strengthening - Stratified Oversampling & Copy-Paste for detector_v5

This script:
1. Copies the entire detector_v4 splits (train, val, test) to detector_v5.
2. Identifies images containing minority classes:
   - 5: emniyet_kemeri_ihlali
   - 6: teknocan
   - 7: bilgisayar
3. Extracts crop library of these classes from the training split.
4. Performs copy-paste augmentation in train split:
   - Generates ~420 images containing 'teknocan' crops to reach 500+ instances.
   - Generates ~360 images containing 'emniyet_kemeri_ihlali' crops to reach 1000+ instances.
   - Generates ~200 images containing 'bilgisayar' crops to boost train count.
5. Performs Albumentations augmentation in val split to boost validation representation:
   - 10 'bilgisayar' val images -> generates 4 augmented versions each (+40 instances, total 50).
   - 20 'teknocan' val images -> generates 2 augmented versions each (+40 instances, total 60).
6. Generates configs/data_v5.yaml and data/curated/detector_v5/data.yaml.
7. Logs split details to reports/detector_v5_split_summary.json.

Usage:
    python scripts/prepare_v5_dataset.py [--dry-run]
"""

import os
import sys
import shutil
import random
import argparse
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import numpy as np
import yaml
import albumentations as A

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("sinaptic5g.prepare_v5")

PROJECT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT / "data/curated/detector_v4"
DST_DIR = PROJECT / "data/curated/detector_v5"

FTR_CLASSES = {
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

# Image helper functions
def read_yolo_label(label_path: Path) -> list:
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4])))
    return rows

def write_yolo_label(label_path: Path, annotations: list) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(a[0])} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}" for a in annotations]
    label_path.write_text("\n".join(lines), encoding="utf-8")

def yolo_to_pixel(cx: float, cy: float, bw: float, bh: float, img_w: int, img_h: int) -> tuple:
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def pixel_to_yolo(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple:
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh

def iou_bbox(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-6)

# Copy-Paste Blend & Positions
def feather_blend(target: np.ndarray, patch: np.ndarray, x1: int, y1: int, feather_px: int = 4) -> np.ndarray:
    h_patch, w_patch = patch.shape[:2]
    x2, y2 = x1 + w_patch, y1 + h_patch
    
    if x2 > target.shape[1] or y2 > target.shape[0]:
        return target
    
    target_region = target[y1:y2, x1:x2]
    
    # Generate linear boundary feather mask
    mask = np.ones((h_patch, w_patch), dtype=np.float32)
    feather = min(feather_px, h_patch // 4, w_patch // 4)
    if feather > 0:
        for i in range(feather):
            alpha = (i + 1) / (feather + 1)
            mask[i, :] = alpha
            mask[-(i + 1), :] = alpha
            mask[:, i] = np.minimum(mask[:, i], alpha)
            mask[:, -(i + 1)] = np.minimum(mask[:, -(i + 1)], alpha)
            
    mask3 = np.stack([mask, mask, mask], axis=-1)
    blended = (patch.astype(np.float32) * mask3 + target_region.astype(np.float32) * (1.0 - mask3))
    
    result = target.copy()
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result

def find_valid_paste_position(target_h: int, target_w: int, patch_h: int, patch_w: int, 
                              existing_bboxes: list, max_iou: float = 0.15, max_attempts: int = 30, 
                              rng: random.Random = None) -> tuple:
    if rng is None:
        rng = random.Random()
    for _ in range(max_attempts):
        margin_x = max(0, target_w - patch_w)
        margin_y = max(0, target_h - patch_h)
        if margin_x == 0 or margin_y == 0:
            return None
        x1 = rng.randint(0, margin_x)
        y1 = rng.randint(0, margin_y)
        x2 = x1 + patch_w
        y2 = y1 + patch_h
        
        candidate = (x1, y1, x2, y2)
        if all(iou_bbox(candidate, ex) < max_iou for ex in existing_bboxes):
            return x1, y1
    return None

def scale_patch(patch: np.ndarray, target_h: int, target_w: int, min_ratio: float = 0.08, max_ratio: float = 0.22, 
                rng: random.Random = None) -> np.ndarray:
    if rng is None:
        rng = random.Random()
    ratio = rng.uniform(min_ratio, max_ratio)
    new_h = int(target_h * ratio)
    new_w = int(patch.shape[1] * new_h / patch.shape[0])
    
    new_h = max(16, min(new_h, target_h - 1))
    new_w = max(16, min(new_w, target_w - 1))
    return cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

def cv2_imread_unicode(path: Path) -> np.ndarray | None:
    try:
        file_bytes = np.fromfile(str(path), dtype=np.uint8)
        if file_bytes.size == 0:
            return None
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        LOG.warning(f"Error reading image {path} with numpy.fromfile: {e}")
        return None

def cv2_imwrite_unicode(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    try:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception as e:
        LOG.warning(f"Error writing image {path} with numpy.tofile: {e}")
        return False

def build_crop_library(images_dir: Path, labels_dir: Path, target_classes: set) -> dict:
    library = {cid: [] for cid in target_classes}
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        annotations = read_yolo_label(label_path)
        if not any(ann[0] in target_classes for ann in annotations):
            continue
            
        img = cv2_imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        for ann in annotations:
            cid, cx, cy, bw, bh = ann
            if cid not in target_classes:
                continue
            x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, w, h)
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 16 or (y2 - y1) < 16:
                continue
            crop = img[y1:y2, x1:x2].copy()
            if crop.size > 0:
                library[cid].append(crop)
    return library

# Albumentations pipelines
def get_val_aug_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=1.0),
        ], p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.4))

def main():
    parser = argparse.ArgumentParser(description="detector_v5 Dataset Builder")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without copying/generating files")
    args = parser.parse_args()
    
    rng = random.Random(42)
    
    if not SRC_DIR.exists():
        LOG.error(f"Source folder {SRC_DIR} does not exist. Ensure detector_v4 exists.")
        sys.exit(1)
        
    LOG.info(f"Preparing detector_v5 dataset. Dry-run status: {args.dry_run}")
    
    splits = ["train", "val", "test"]
    
    # 1. Verify and create directories
    if not args.dry_run:
        DST_DIR.mkdir(parents=True, exist_ok=True)
        for split in splits:
            (DST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
            (DST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
            
    # 2. Copy original dataset files
    original_counts = {split: 0 for split in splits}
    for split in splits:
        src_images = sorted(list((SRC_DIR / split / "images").glob("*.jpg")) + list((SRC_DIR / split / "images").glob("*.png")))
        original_counts[split] = len(src_images)
        LOG.info(f"Copying {len(src_images)} original images and labels for split '{split}'...")
        if not args.dry_run:
            for img_path in src_images:
                lbl_path = SRC_DIR / split / "labels" / (img_path.stem + ".txt")
                shutil.copy2(img_path, DST_DIR / split / "images" / img_path.name)
                if lbl_path.exists():
                    shutil.copy2(lbl_path, DST_DIR / split / "labels" / lbl_path.name)

    # 3. Build Crop Library from original Train Split
    LOG.info("Building crop library for classes 5, 6, 7 from train split...")
    crop_library = build_crop_library(SRC_DIR / "train/images", SRC_DIR / "train/labels", {5, 6, 7})
    for cid, crops in crop_library.items():
        LOG.info(f"  Class {cid} ({FTR_CLASSES[cid]}): {len(crops)} crops extracted.")
        
    # 4. Train Copy-Paste Augmentations
    # Let's specify exact targets:
    # We want 500+ teknocan (class 6), starting from 90. Needs +420 cp.
    # We want 1000+ emniyet_kemeri (class 5), starting from 654. Needs +360 cp.
    # We want to increase bilgisayar (class 7), starting from 335. Let's do +200 cp.
    cp_targets = [
        (6, 420),  # class 6, 420 copies
        (5, 360),  # class 5, 360 copies
        (7, 200),  # class 7, 200 copies
    ]
    
    train_image_paths = sorted(list((SRC_DIR / "train/images").glob("*.jpg")) + list((SRC_DIR / "train/images").glob("*.png")))
    
    cp_counter = 0
    if not args.dry_run:
        for cid, num_copies in cp_targets:
            LOG.info(f"Generating {num_copies} copy-paste images for Class {cid} ({FTR_CLASSES[cid]})...")
            crops = crop_library[cid]
            if not crops:
                LOG.warning(f"No crops available for Class {cid}. Skipping.")
                continue
                
            for _ in range(num_copies):
                # Pick a random training image as background
                bg_img_path = rng.choice(train_image_paths)
                bg_lbl_path = SRC_DIR / "train/labels" / (bg_img_path.stem + ".txt")
                
                bg_img = cv2_imread_unicode(bg_img_path)
                if bg_img is None:
                    continue
                h, w = bg_img.shape[:2]
                annotations = read_yolo_label(bg_lbl_path)
                existing_bboxes = [yolo_to_pixel(a[1], a[2], a[3], a[4], w, h) for a in annotations]
                
                # Copy-paste 1 or 2 patches
                n_patches = rng.choice([1, 2])
                aug_img = bg_img.copy()
                aug_anns = list(annotations)
                
                success = False
                for _ in range(n_patches):
                    patch = rng.choice(crops)
                    # Color jitter
                    jitter = rng.uniform(0.85, 1.15)
                    patch_jitter = np.clip(patch.astype(np.float32) * jitter, 0, 255).astype(np.uint8)
                    
                    # Scale patch
                    scaled = scale_patch(patch_jitter, h, w, rng=rng)
                    ph, pw = scaled.shape[:2]
                    
                    pos = find_valid_paste_position(h, w, ph, pw, existing_bboxes, rng=rng)
                    if pos is not None:
                        px1, py1 = pos
                        px2, py2 = px1 + pw, py1 + ph
                        aug_img = feather_blend(aug_img, scaled, px1, py1, feather_px=3)
                        
                        cx_n, cy_n, bw_n, bh_n = pixel_to_yolo(px1, py1, px2, py2, w, h)
                        aug_anns.append((cid, cx_n, cy_n, bw_n, bh_n))
                        existing_bboxes.append((px1, py1, px2, py2))
                        success = True
                        
                if success:
                    out_img_name = f"cp_v5_aug_{cp_counter:05d}.jpg"
                    out_lbl_name = f"cp_v5_aug_{cp_counter:05d}.txt"
                    
                    cv2_imwrite_unicode(DST_DIR / "train/images" / out_img_name, aug_img, 95)
                    write_yolo_label(DST_DIR / "train/labels" / out_lbl_name, aug_anns)
                    cp_counter += 1
                    
        LOG.info(f"Completed Train Copy-Paste. Generated {cp_counter} new train images.")
    else:
        LOG.info(f"Dry-run: Would generate 980 copy-paste images in train.")
        
    # 5. Validation Augmentations (without leaking train samples)
    # val bilgisayar (class 7): 10 val images -> generate 4 augmented versions each
    # val teknocan (class 6): 20 val images -> generate 2 augmented versions each
    val_aug_pipeline = get_val_aug_pipeline()
    val_image_paths = sorted(list((SRC_DIR / "val/images").glob("*.jpg")) + list((SRC_DIR / "val/images").glob("*.png")))
    
    val_aug_counter = 0
    if not args.dry_run:
        LOG.info("Augmenting validation minority class samples to avoid validation sparsity...")
        for val_img_path in val_image_paths:
            val_lbl_path = SRC_DIR / "val/labels" / (val_img_path.stem + ".txt")
            annotations = read_yolo_label(val_lbl_path)
            
            has_bilgisayar = any(ann[0] == 7 for ann in annotations)
            has_teknocan = any(ann[0] == 6 for ann in annotations)
            
            num_augs = 0
            if has_bilgisayar:
                num_augs = 4
            elif has_teknocan:
                num_augs = 2
                
            if num_augs > 0:
                img = cv2_imread_unicode(val_img_path)
                if img is None:
                    continue
                    
                for aug_idx in range(num_augs):
                    # albumentations bbox format: [x_min, y_min, width, height] normalized
                    bboxes = [(a[1], a[2], a[3], a[4]) for a in annotations]
                    class_labels = [a[0] for a in annotations]
                    
                    try:
                        res = val_aug_pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
                        aug_img = res["image"]
                        aug_bboxes = res["bboxes"]
                        aug_labels = res["class_labels"]
                        
                        if len(aug_bboxes) > 0:
                            out_img_name = f"val_aug_{val_aug_counter:05d}.jpg"
                            out_lbl_name = f"val_aug_{val_aug_counter:05d}.txt"
                            
                            cv2_imwrite_unicode(DST_DIR / "val/images" / out_img_name, aug_img, 95)
                            
                            new_anns = [(aug_labels[i], aug_bboxes[i][0], aug_bboxes[i][1], aug_bboxes[i][2], aug_bboxes[i][3])
                                        for i in range(len(aug_bboxes))]
                            write_yolo_label(DST_DIR / "val/labels" / out_lbl_name, new_anns)
                            val_aug_counter += 1
                    except Exception as e:
                        LOG.debug(f"Validation augmentation failed: {e}")
                        
        LOG.info(f"Completed Validation Augmentation. Generated {val_aug_counter} new validation images.")
    else:
        LOG.info("Dry-run: Would generate augmented images in val (4 per val_bilgisayar, 2 per val_teknocan).")
        
    # 6. Analyze and verify dataset split summary
    LOG.info("Running class verification on prepared detector_v5 dataset...")
    
    split_summary = {}
    for split in splits:
        if args.dry_run:
            # For dry-run, report source counts as a placeholder
            img_dir = SRC_DIR / split / "images"
            lbl_dir = SRC_DIR / split / "labels"
        else:
            img_dir = DST_DIR / split / "images"
            lbl_dir = DST_DIR / split / "labels"
            
        images = list(img_dir.glob("*"))
        labels = list(lbl_dir.glob("*.txt"))
        
        instances = Counter()
        for lbl in labels:
            for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                        instances[cls_id] += 1
                    except ValueError:
                        continue
                        
        split_summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "instances": {str(k): v for k, v in sorted(instances.items())}
        }
        
    LOG.info("--- detector_v5 Dataset Summary ---")
    for split, details in split_summary.items():
        LOG.info(f"Split {split}: {details['images']} images")
        for cid_str, cnt in sorted(details['instances'].items(), key=lambda x: int(x[0])):
            LOG.info(f"  Class {cid_str} ({FTR_CLASSES[int(cid_str)]}): {cnt} instances")
            
    # Save the split summary to reports directory
    if not args.dry_run:
        reports_dir = PROJECT / "reports"
        reports_dir.mkdir(exist_ok=True)
        (reports_dir / "detector_v5_split_summary.json").write_text(
            json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOG.info(f"Saved detector_v5_split_summary.json to {reports_dir}")
        
        # Save a comparison CSV
        csv_lines = ["class_id,class_name,v4_train_support,v5_train_support,v4_val_support,v5_val_support"]
        v4_summary_path = reports_dir / "detector_v4_split_summary.json"
        if v4_summary_path.is_file():
            v4_summary = json.loads(v4_summary_path.read_text(encoding="utf-8"))
            for cid in sorted(FTR_CLASSES.keys()):
                v4_train = v4_summary.get("train", {}).get("instances", {}).get(str(cid), 0)
                v5_train = split_summary.get("train", {}).get("instances", {}).get(str(cid), 0)
                v4_val = v4_summary.get("val", {}).get("instances", {}).get(str(cid), 0)
                v5_val = split_summary.get("val", {}).get("instances", {}).get(str(cid), 0)
                csv_lines.append(f"{cid},{FTR_CLASSES[cid]},{v4_train},{v5_train},{v4_val},{v5_val}")
                
            (reports_dir / "detector_v5_class_comparison.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
            LOG.info(f"Saved detector_v5_class_comparison.csv to {reports_dir}")
            
        # Write data/curated/detector_v5/data.yaml
        data_yaml_path = DST_DIR / "data.yaml"
        yaml_content = {
            "path": str(DST_DIR.resolve()).replace("\\", "/"),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {int(k): v for k, v in FTR_CLASSES.items()},
            "nc": len(FTR_CLASSES)
        }
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False, allow_unicode=True)
        LOG.info(f"Saved {data_yaml_path}")
        
        # Also copy it to configs/data_v5.yaml as a shortcut
        shutil.copy2(data_yaml_path, PROJECT / "configs/data_v5.yaml")
        LOG.info(f"Saved configs/data_v5.yaml")
        
    return 0

if __name__ == "__main__":
    main()
