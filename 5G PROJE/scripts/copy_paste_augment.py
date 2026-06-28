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
scripts/copy_paste_augment.py — Copy-Paste Augmentasyon Altyapısı
==================================================================
Faz 2: Veri Seti Güçlendirme — Copy-Paste Stratejisi

Segment/mask yoksa bbox tabanlı paste (fallback) kullanır.
Düşük destekli sınıflar (teknocan, bilgisayar, emniyet_kemeri_ihlali)
için hedef görüntüye objeler yapıştırılır.

Kullanım:
    python scripts/copy_paste_augment.py \
        --train-images dataset/images/train \
        --train-labels dataset/labels/train \
        --output-images data/augmented/images/train \
        --output-labels data/augmented/labels/train \
        --paste-classes 5 6 7 \
        --n-paste 2 \
        --n-output 500

Üretim kilidi: detector_v3 / detector.onnx KESİNLİKLE dokunulmaz.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOG = logging.getLogger("sinaptic5g.copy_paste")

# FTR sınıf haritası
FTR_CLASS_MAP: Dict[int, str] = {
    0: "telefonla_konusma",
    1: "su_icme",
    2: "arkaya_bakma",
    3: "esneme",
    4: "sigara_icme",
    5: "emniyet_kemeri_ihlali",
    6: "teknocan",
    7: "bilgisayar",
    8: "license_plate",
}


def read_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """YOLO etiket dosyasını okur. [(class_id, cx, cy, w, h)]"""
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4])))
    return rows


def write_yolo_label(label_path: Path, annotations: List[Tuple]) -> None:
    """YOLO etiket dosyası yazar."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{a[0]} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}" for a in annotations]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def yolo_to_pixel(cx: float, cy: float, bw: float, bh: float,
                   img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """YOLO normalize → piksel bbox [x1, y1, x2, y2]."""
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def pixel_to_yolo(x1: int, y1: int, x2: int, y2: int,
                   img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Piksel bbox → YOLO normalize [cx, cy, w, h]."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def iou_bbox(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """IoU hesapla."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def build_source_library(
    images_dir: Path,
    labels_dir: Path,
    paste_classes: set,
) -> Dict[int, List[Tuple[np.ndarray, int]]]:
    """
    Yapıştırılacak nesnelerin kütüphanesini oluşturur.
    Returns: {class_id: [(crop_img, class_id), ...]}
    """
    library: Dict[int, List[Tuple[np.ndarray, int]]] = {cid: [] for cid in paste_classes}
    
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        annotations = read_yolo_label(label_path)
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        for ann in annotations:
            cid, cx, cy, bw, bh = ann
            if cid not in paste_classes:
                continue
            
            x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, w, h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 16 or (y2 - y1) < 16:
                continue
            
            crop = img[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            
            library[cid].append((crop, cid))
    
    for cid, items in library.items():
        LOG.info("Kütüphane: class=%s (%s), %d nesne",
                  cid, FTR_CLASS_MAP.get(cid, "?"), len(items))
    
    return library


def feather_blend(
    target: np.ndarray,
    patch: np.ndarray,
    x1: int, y1: int,
    feather_px: int = 4,
) -> np.ndarray:
    """
    Patch'i target'a yumuşak kenar (feathering) ile yapıştırır.
    Segment/mask yoksa bbox tabanlı paste kullanılır (fallback).
    """
    h_patch, w_patch = patch.shape[:2]
    x2, y2 = x1 + w_patch, y1 + h_patch
    
    # Sınır kontrolü
    if x2 > target.shape[1] or y2 > target.shape[0]:
        return target
    
    # Basit bbox paste (fallback — mask yok)
    target_region = target[y1:y2, x1:x2]
    
    # Feather mask oluştur
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
    blended = (patch.astype(np.float32) * mask3 +
               target_region.astype(np.float32) * (1.0 - mask3))
    
    result = target.copy()
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def find_valid_paste_position(
    target_h: int, target_w: int,
    patch_h: int, patch_w: int,
    existing_bboxes: List[Tuple[int, int, int, int]],
    max_iou: float = 0.2,
    max_attempts: int = 20,
    rng: Optional[random.Random] = None,
) -> Optional[Tuple[int, int]]:
    """Çakışmayan geçerli bir yapıştırma konumu bulur."""
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
        
        # IoU kontrolü
        ok = all(iou_bbox(candidate, ex) < max_iou for ex in existing_bboxes)
        if ok:
            return x1, y1
    
    return None


def scale_patch(
    patch: np.ndarray,
    target_h: int,
    target_w: int,
    min_ratio: float = 0.06,
    max_ratio: float = 0.25,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Patch'i hedef görüntüye göre makul ölçekte yeniden boyutlandırır."""
    if rng is None:
        rng = random.Random()
    
    ratio = rng.uniform(min_ratio, max_ratio)
    new_h = int(target_h * ratio)
    new_w = int(patch.shape[1] * new_h / patch.shape[0])
    
    # Minimum boyut garantisi
    new_h = max(new_h, 16)
    new_w = max(new_w, 16)
    new_h = min(new_h, target_h - 1)
    new_w = min(new_w, target_w - 1)
    
    return cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def copy_paste_one(
    target_img: np.ndarray,
    target_anns: List[Tuple[int, float, float, float, float]],
    library: Dict[int, List[Tuple[np.ndarray, int]]],
    paste_classes: set,
    n_paste: int = 2,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """
    Bir görüntüye copy-paste augmentasyon uygular.
    Returns: (augmented_img, updated_annotations)
    """
    if rng is None:
        rng = random.Random()
    
    h, w = target_img.shape[:2]
    
    # Mevcut piksel bbox'ları
    existing_bboxes = [yolo_to_pixel(a[1], a[2], a[3], a[4], w, h) for a in target_anns]
    result_img = target_img.copy()
    result_anns = list(target_anns)
    
    # Rastgele sınıf sırası
    available_classes = [cid for cid in paste_classes if library.get(cid)]
    if not available_classes:
        return result_img, result_anns
    
    rng.shuffle(available_classes)
    
    pasted_count = 0
    for cid in available_classes * ((n_paste // len(available_classes)) + 1):
        if pasted_count >= n_paste:
            break
        
        items = library.get(cid)
        if not items:
            continue
        
        patch_img, patch_cid = rng.choice(items)
        
        # Hafif renk jitter
        jitter = rng.uniform(0.85, 1.15)
        patch_aug = np.clip(patch_img.astype(np.float32) * jitter, 0, 255).astype(np.uint8)
        
        # Ölçeklendir
        scaled_patch = scale_patch(patch_aug, h, w, rng=rng)
        ph, pw = scaled_patch.shape[:2]
        
        # Konum bul
        pos = find_valid_paste_position(h, w, ph, pw, existing_bboxes, rng=rng)
        if pos is None:
            continue
        
        px1, py1 = pos
        px2, py2 = px1 + pw, py1 + ph
        
        # Yapıştır
        result_img = feather_blend(result_img, scaled_patch, px1, py1, feather_px=3)
        
        # Yeni bbox ekle
        cx, cy, bw_norm, bh_norm = pixel_to_yolo(px1, py1, px2, py2, w, h)
        result_anns.append((patch_cid, cx, cy, bw_norm, bh_norm))
        existing_bboxes.append((px1, py1, px2, py2))
        pasted_count += 1
    
    return result_img, result_anns


def run_copy_paste(
    train_images_dir: Path,
    train_labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    paste_classes: set,
    n_paste: int = 2,
    n_output: int = 500,
    seed: int = 42,
) -> Dict:
    """Ana copy-paste augmentasyon fonksiyonu."""
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    rng = random.Random(seed)
    np.random.seed(seed)
    
    # Kütüphane oluştur
    LOG.info("Nesne kütüphanesi oluşturuluyor...")
    library = build_source_library(train_images_dir, train_labels_dir, paste_classes)
    
    total_objects = sum(len(v) for v in library.values())
    if total_objects == 0:
        LOG.warning("Düşük destekli sınıflar için kütüphane boş!")
        return {"error": "empty_library", "paste_classes": list(paste_classes)}
    
    image_paths = sorted(list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png")))
    
    if not image_paths:
        return {"error": "no_images"}
    
    generated = 0
    stats: Dict = {
        "library_objects": {FTR_CLASS_MAP.get(cid, str(cid)): len(v) for cid, v in library.items()},
        "generated": 0,
        "skipped": 0,
    }
    
    while generated < n_output:
        img_path = rng.choice(image_paths)
        label_path = train_labels_dir / (img_path.stem + ".txt")
        
        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped"] += 1
            continue
        
        annotations = read_yolo_label(label_path)
        
        aug_img, aug_anns = copy_paste_one(
            img, annotations, library, paste_classes, n_paste=n_paste, rng=rng
        )
        
        stem = f"cp_aug_{generated:05d}"
        out_img_path = output_images_dir / f"{stem}.jpg"
        out_lbl_path = output_labels_dir / f"{stem}.txt"
        
        cv2.imwrite(str(out_img_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_yolo_label(out_lbl_path, aug_anns)
        
        generated += 1
        
        if generated % 50 == 0:
            LOG.info("Copy-paste: %d/%d görüntü oluşturuldu", generated, n_output)
    
    stats["generated"] = generated
    LOG.info("Copy-paste augmentasyon tamamlandı: %d görüntü", generated)
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Sinaptic5G — Copy-Paste Augmentasyon")
    parser.add_argument("--train-images", type=Path, default=Path("dataset/images/train"))
    parser.add_argument("--train-labels", type=Path, default=Path("dataset/labels/train"))
    parser.add_argument("--output-images", type=Path, default=Path("data/augmented/images/train"))
    parser.add_argument("--output-labels", type=Path, default=Path("data/augmented/labels/train"))
    parser.add_argument("--paste-classes", type=int, nargs="+", default=[5, 6, 7],
                        help="Yapıştırılacak sınıf ID'leri (5=emniyet_kemeri, 6=teknocan, 7=bilgisayar)")
    parser.add_argument("--n-paste", type=int, default=2,
                        help="Her görüntüye kaç nesne yapıştırılacak")
    parser.add_argument("--n-output", type=int, default=500,
                        help="Kaç augmented görüntü üretilecek")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    LOG.info("Copy-paste augmentasyon başlıyor...")
    LOG.info("  Paste sınıfları: %s", [FTR_CLASS_MAP.get(c, str(c)) for c in args.paste_classes])
    LOG.info("  Hedef görüntü sayısı: %d", args.n_output)
    
    stats = run_copy_paste(
        train_images_dir=args.train_images,
        train_labels_dir=args.train_labels,
        output_images_dir=args.output_images,
        output_labels_dir=args.output_labels,
        paste_classes=set(args.paste_classes),
        n_paste=args.n_paste,
        n_output=args.n_output,
        seed=args.seed,
    )
    
    if "error" in stats:
        LOG.error("Hata: %s", stats)
        return 1
    
    LOG.info("Özet: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
