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

"""Extract transparent background Teknocan mascot foreground images from user-uploaded images."""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path

def cv2_imread_unicode(path: Path) -> np.ndarray | None:
    try:
        file_bytes = np.fromfile(str(path), dtype=np.uint8)
        if file_bytes.size == 0:
            return None
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def cv2_imwrite_unicode(path: Path, img: np.ndarray, is_png: bool = True) -> bool:
    try:
        ext = '.png' if is_png else '.jpg'
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception as e:
        print(f"Error writing {path}: {e}")
        return False

def remove_white_background(img: np.ndarray, thresh: int = 240) -> np.ndarray:
    """Convert white/near-white background to transparent."""
    # Create mask of non-white pixels
    mask = (img[:, :, 0] < thresh) | (img[:, :, 1] < thresh) | (img[:, :, 2] < thresh)
    
    # Crop to bounding box of mask
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    cropped_img = img[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]
    
    # Convert to BGRA
    bgra = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = np.where(cropped_mask, 255, 0).astype(np.uint8)
    
    # Smooth edges with a small blur on the alpha channel
    alpha = bgra[:, :, 3]
    alpha_blurred = cv2.GaussianBlur(alpha, (3, 3), 0)
    bgra[:, :, 3] = np.where(alpha > 0, alpha_blurred, 0)
    
    return bgra

def main():
    workspace_root = Path(__file__).resolve().parents[1]
    
    # Source paths in the conversation brain
    brain_dir = Path("C:/Users/seydieryilmaz/.gemini/antigravity-ide/brain/20ec6ad0-9b96-44fc-b494-abde89a531fe")
    img1_src = brain_dir / "media__1782584343122.png"
    img2_src = brain_dir / "media__1782584521455.png"
    
    # Create raw source directory if it doesn't exist
    source_dir = workspace_root / "data/raw/teknocan_source_images"
    source_dir.mkdir(parents=True, exist_ok=True)
    
    # Target directory for foregrounds
    fg_dir = workspace_root / "data/raw/teknocan_fg"
    fg_dir.mkdir(parents=True, exist_ok=True)
    
    print("Copying source images to workspace...")
    if img1_src.is_file():
        shutil.copy2(img1_src, source_dir / "teknocan_1.png")
    if img2_src.is_file():
        shutil.copy2(img2_src, source_dir / "teknocan_2.png")
        
    local_img1 = source_dir / "teknocan_1.png"
    local_img2 = source_dir / "teknocan_2.png"
    
    if not local_img1.is_file() or not local_img2.is_file():
        print("Error: Source images not found.")
        return 1
        
    # Process image 1 (single mascot)
    print("Processing image 1 (single mascot)...")
    img1 = cv2_imread_unicode(local_img1)
    if img1 is not None:
        rgba1 = remove_white_background(img1, thresh=245)
        out_path = fg_dir / "teknocan_mascot_1.png"
        cv2_imwrite_unicode(out_path, rgba1, is_png=True)
        print(f"Saved {out_path}")
        
    # Process image 2 (multiple illustrations sheet)
    print("Processing image 2 (illustrations sheet)...")
    img2 = cv2_imread_unicode(local_img2)
    if img2 is not None:
        # Segment into multiple mascots
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # Threshold to find non-white regions
        _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        idx = 2
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 8000: # Filter out small noise
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            # Add padding
            padding = 10
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(img2.shape[1] - x_pad, w + 2 * padding)
            h_pad = min(img2.shape[0] - y_pad, h + 2 * padding)
            
            crop = img2[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            rgba_crop = remove_white_background(crop, thresh=245)
            
            out_path = fg_dir / f"teknocan_mascot_{idx}.png"
            cv2_imwrite_unicode(out_path, rgba_crop, is_png=True)
            print(f"Saved mascot contour {i} (area: {area}) as {out_path}")
            idx += 1
            
    print("Teknocan foreground extraction complete.")
    return 0

if __name__ == "__main__":
    main()
