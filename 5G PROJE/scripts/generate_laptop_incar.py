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

"""Generate in-car laptop images by copy-pasting laptop crops onto cabin backgrounds."""

import os
import random
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np

def extract_laptop_crops(laptop_dir):
    crops = []
    # Search in all splits (train, valid, test)
    for split in ["train", "valid", "test"]:
        labels_dir = laptop_dir / split / "labels"
        images_dir = laptop_dir / split / "images"
        if not labels_dir.is_dir():
            continue
        
        for label_path in labels_dir.glob("*.txt"):
            image_path = images_dir / f"{label_path.stem}.jpg"
            if not image_path.is_file():
                image_path = images_dir / f"{label_path.stem}.png"
                if not image_path.is_file():
                    image_path = images_dir / f"{label_path.stem}.jpeg"
                    if not image_path.is_file():
                        continue
            
            img = cv2.imread(str(image_path))
            if img is None:
                continue
                
            h_img, w_img, _ = img.shape
            try:
                content = label_path.read_text(encoding="utf-8")
            except Exception:
                continue
                
            for line in content.splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                    if cid == 0:  # laptop class in raw dataset
                        cx, cy, w, h = [float(x) for x in parts[1:5]]
                        x_min = int((cx - w/2) * w_img)
                        y_min = int((cy - h/2) * h_img)
                        crop_w = int(w * w_img)
                        crop_h = int(h * h_img)
                        
                        # Clip coords
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        crop_w = min(w_img - x_min, crop_w)
                        crop_h = min(h_img - y_min, crop_h)
                        
                        if crop_w > 10 and crop_h > 10:
                            crop = img[y_min:y_min+crop_h, x_min:x_min+crop_w]
                            crops.append(crop)
                except Exception:
                    pass
    return crops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laptop-dir", type=Path, default=Path("data/raw/detect_laptop"))
    parser.add_argument("--bg-dir", type=Path, default=Path("data/raw/cabin_backgrounds"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/laptop_incar"))
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()

    # Verify input directories
    if not args.laptop_dir.is_dir():
        print(f"Error: Laptop directory not found: {args.laptop_dir}")
        return 1
    if not args.bg_dir.is_dir():
        print(f"Error: Background directory not found: {args.bg_dir}")
        return 1

    # Extract crops
    print("Extracting laptop crops...")
    crops = extract_laptop_crops(args.laptop_dir)
    print(f"Extracted {len(crops)} laptop crops.")
    if not crops:
        print("Error: No laptop crops could be extracted.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "images").mkdir(exist_ok=True)
    (args.output_dir / "labels").mkdir(exist_ok=True)

    bg_images = list(args.bg_dir.glob("*.jpg")) + list(args.bg_dir.glob("*.png")) + list(args.bg_dir.glob("*.jpeg"))
    if not bg_images:
        print("Error: No background images found.")
        return 1

    generated = 0
    while generated < args.count:
        bg_path = random.choice(bg_images)
        fg_crop = random.choice(crops)

        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue

        bg_h, bg_w, _ = bg.shape

        # Scale crop (0.10 to 0.30 of image area)
        target_area = bg_h * bg_w * random.uniform(0.10, 0.30)
        fg_aspect = fg_crop.shape[1] / fg_crop.shape[0]
        new_h = int(np.sqrt(target_area / fg_aspect))
        new_w = int(new_h * fg_aspect)

        if new_h == 0 or new_w == 0 or new_h >= bg_h or new_w >= bg_w:
            continue

        fg_scaled = cv2.resize(fg_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Random horizontal flip (prob: 0.5)
        if random.random() < 0.5:
            fg_scaled = cv2.flip(fg_scaled, 1)

        # Random brightness/contrast
        alpha = random.uniform(0.8, 1.2) # contrast
        beta = random.randint(-20, 20)  # brightness
        fg_final = cv2.convertScaleAbs(fg_scaled, alpha=alpha, beta=beta)

        # Random location in the lower half (simulating steering wheel, lap, console)
        y_min = int(bg_h / 2)
        y_max = bg_h - new_h - 1
        x_max = bg_w - new_w - 1

        if y_max <= y_min or x_max <= 0:
            continue

        y_start = random.randint(y_min, y_max)
        x_start = random.randint(0, x_max)

        # Paste (regular copy-paste without alpha channel as laptops are cropped from JPG)
        bg[y_start:y_start+new_h, x_start:x_start+new_w] = fg_final

        # YOLO annotation parameters (class_id 7)
        cx = (x_start + new_w / 2.0) / bg_w
        cy = (y_start + new_h / 2.0) / bg_h
        w = new_w / bg_w
        h = new_h / bg_h

        img_name = f"laptop_incar_{generated:04d}.jpg"
        lbl_name = f"laptop_incar_{generated:04d}.txt"

        cv2.imwrite(str(args.output_dir / "images" / img_name), bg)
        with (args.output_dir / "labels" / lbl_name).open("w", encoding="utf-8") as f:
            f.write(f"7 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        generated += 1

    print(f"Successfully generated {generated} in-car laptop images.")
    return 0

if __name__ == "__main__":
    main()
