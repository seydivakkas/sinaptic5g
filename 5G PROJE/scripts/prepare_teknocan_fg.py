# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

"""Extract transparent-background Teknocan foreground images from approved local sources.

Source images are expected in ``data/raw/teknocan_source_images`` by default.
Set ``SINAPTIC5G_TEKNOCAN_SOURCE_DIR`` to use another local directory without
hard-coding developer-specific filesystem paths into the repository.
"""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def cv2_imread_unicode(path: Path) -> np.ndarray | None:
    try:
        file_bytes = np.fromfile(str(path), dtype=np.uint8)
        if file_bytes.size == 0:
            return None
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as exc:
        print(f"Error reading {path}: {exc}")
        return None


def cv2_imwrite_unicode(path: Path, img: np.ndarray, is_png: bool = True) -> bool:
    try:
        ext = ".png" if is_png else ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception as exc:
        print(f"Error writing {path}: {exc}")
        return False


def remove_white_background(img: np.ndarray, thresh: int = 240) -> np.ndarray:
    """Convert white/near-white background to transparent."""
    mask = (img[:, :, 0] < thresh) | (img[:, :, 1] < thresh) | (img[:, :, 2] < thresh)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped_img = img[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]

    bgra = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = np.where(cropped_mask, 255, 0).astype(np.uint8)

    alpha = bgra[:, :, 3]
    alpha_blurred = cv2.GaussianBlur(alpha, (3, 3), 0)
    bgra[:, :, 3] = np.where(alpha > 0, alpha_blurred, 0)
    return bgra


def _prepare_sources(workspace_root: Path) -> tuple[Path, Path] | None:
    source_dir = workspace_root / "data/raw/teknocan_source_images"
    source_dir.mkdir(parents=True, exist_ok=True)

    external_value = os.getenv("SINAPTIC5G_TEKNOCAN_SOURCE_DIR", "").strip()
    if external_value:
        external_dir = Path(external_value).expanduser().resolve()
        if not external_dir.is_dir():
            print(f"Error: SINAPTIC5G_TEKNOCAN_SOURCE_DIR is not a directory: {external_dir}")
            return None

        candidates = sorted(
            path
            for path in external_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if len(candidates) < 2:
            print("Error: At least two approved Teknocan source images are required.")
            return None

        shutil.copy2(candidates[0], source_dir / "teknocan_1.png")
        shutil.copy2(candidates[1], source_dir / "teknocan_2.png")

    img1 = source_dir / "teknocan_1.png"
    img2 = source_dir / "teknocan_2.png"
    if not img1.is_file() or not img2.is_file():
        print(
            "Error: Approved source images not found. Place teknocan_1.png and "
            "teknocan_2.png under data/raw/teknocan_source_images or set "
            "SINAPTIC5G_TEKNOCAN_SOURCE_DIR."
        )
        return None
    return img1, img2


def main() -> int:
    workspace_root = Path(__file__).resolve().parents[1]
    prepared = _prepare_sources(workspace_root)
    if prepared is None:
        return 1
    local_img1, local_img2 = prepared

    fg_dir = workspace_root / "data/raw/teknocan_fg"
    fg_dir.mkdir(parents=True, exist_ok=True)

    print("Processing image 1 (single mascot)...")
    img1 = cv2_imread_unicode(local_img1)
    if img1 is not None:
        rgba1 = remove_white_background(img1, thresh=245)
        out_path = fg_dir / "teknocan_mascot_1.png"
        cv2_imwrite_unicode(out_path, rgba1, is_png=True)
        print(f"Saved {out_path}")

    print("Processing image 2 (illustrations sheet)...")
    img2 = cv2_imread_unicode(local_img2)
    if img2 is not None:
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        idx = 2
        for contour_index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 8000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            padding = 10
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(img2.shape[1] - x_pad, w + 2 * padding)
            h_pad = min(img2.shape[0] - y_pad, h + 2 * padding)

            crop = img2[y_pad : y_pad + h_pad, x_pad : x_pad + w_pad]
            rgba_crop = remove_white_background(crop, thresh=245)
            out_path = fg_dir / f"teknocan_mascot_{idx}.png"
            cv2_imwrite_unicode(out_path, rgba_crop, is_png=True)
            print(f"Saved mascot contour {contour_index} (area: {area}) as {out_path}")
            idx += 1

    print("Teknocan foreground extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
