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

"""Generate synthetic teknocan images using copy-paste onto cabin backgrounds."""

import os
import random
import argparse
from pathlib import Path
import cv2
import numpy as np

def cv2_imread_unicode(path: Path, unchanged: bool = False) -> np.ndarray | None:
    try:
        file_bytes = np.fromfile(str(path), dtype=np.uint8)
        if file_bytes.size == 0:
            return None
        flags = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
        return cv2.imdecode(file_bytes, flags)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def cv2_imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    try:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception as e:
        print(f"Error writing {path}: {e}")
        return False

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return result

def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int32) * factor, 0, 255).astype(np.uint8)
    hsv_new = cv2.merge((h, s, v))
    image[:, :, :3] = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg-dir", type=Path, default=Path("data/raw/cabin_backgrounds"))
    parser.add_argument("--fg-dir", type=Path, default=Path("data/raw/teknocan_fg"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/teknocan_synthetic"))
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    # Check if fg-dir is empty or does not exist
    fg_images = []
    if args.fg_dir.is_dir():
        fg_images = list(args.fg_dir.glob("*.png"))

    if not fg_images:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        blocked_file = reports_dir / "teknocan_synthetic_blocked.txt"
        blocked_file.write_text("Teknocan sentetik üretimi engellendi: ön plan görüntüsü yok\n", encoding="utf-8")
        print("Blocked: No foreground images found. Block log written.")
        return 0

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
        fg_path = random.choice(fg_images)

        bg = cv2_imread_unicode(bg_path)
        fg = cv2_imread_unicode(fg_path, unchanged=True) # RGBA

        if bg is None or fg is None:
            continue

        bg_h, bg_w, _ = bg.shape

        # Generate 3 augmented variants for this combination
        for variant in range(3):
            if generated >= args.count:
                break

            # 1. Random scale (0.05 to 0.20 of image area)
            target_area = bg_h * bg_w * random.uniform(0.05, 0.20)
            fg_aspect = fg.shape[1] / fg.shape[0]
            new_h = int(np.sqrt(target_area / fg_aspect))
            new_w = int(new_h * fg_aspect)
            
            if new_h == 0 or new_w == 0:
                continue

            fg_scaled = cv2.resize(fg, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 2. Random rotation (±15 degrees)
            angle = random.uniform(-15, 15)
            fg_rotated = rotate_image(fg_scaled, angle)

            # 3. Random brightness (±20%)
            brightness_factor = random.uniform(0.8, 1.2)
            fg_final = adjust_brightness(fg_rotated, brightness_factor)

            # 4. Random location (dashboard/lower half preferred)
            y_start = random.randint(int(bg_h / 2), bg_h - new_h - 1) if bg_h - new_h - 1 > int(bg_h / 2) else int(bg_h / 2)
            x_start = random.randint(0, bg_w - new_w - 1) if bg_w - new_w - 1 > 0 else 0

            # Calculate actual pasted dimensions (taking boundaries into account)
            paste_h = min(new_h, bg_h - y_start)
            paste_w = min(new_w, bg_w - x_start)
            
            if paste_h <= 0 or paste_w <= 0:
                continue
                
            # Crop fg_final and alpha to match actual paste area
            fg_crop = fg_final[:paste_h, :paste_w]
            alpha = fg_crop[:, :, 3] / 255.0

            # Alpha blending
            for c in range(0, 3):
                bg[y_start:y_start+paste_h, x_start:x_start+paste_w, c] = (
                    alpha * fg_crop[:, :, c] + (1.0 - alpha) * bg[y_start:y_start+paste_h, x_start:x_start+paste_w, c]
                )

            # YOLO annotation parameters
            cx = (x_start + paste_w / 2.0) / bg_w
            cy = (y_start + paste_h / 2.0) / bg_h
            w = paste_w / bg_w
            h = paste_h / bg_h

            # Save
            img_name = f"synth_teknocan_{generated:04d}.jpg"
            lbl_name = f"synth_teknocan_{generated:04d}.txt"

            cv2_imwrite_unicode(args.output_dir / "images" / img_name, bg)
            with (args.output_dir / "labels" / lbl_name).open("w", encoding="utf-8") as f:
                f.write(f"6 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            generated += 1

    print(f"Generated {generated} synthetic images.")
    return 0

if __name__ == "__main__":
    main()
