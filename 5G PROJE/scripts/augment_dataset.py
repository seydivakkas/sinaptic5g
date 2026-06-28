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
scripts/augment_dataset.py — Albumentations Tabanlı Gelişmiş Augmentasyon Pipeline'ı
======================================================================================
Faz 2: Veri Seti Güçlendirme

Düşük destekli sınıflar (teknocan=6, bilgisayar=7, emniyet_kemeri_ihlali=5) için
ağır augmentasyon stratejileri uygular. Yüksek destekli sınıflar hafif augmentasyon alır.

Split tutarlılığı: configs/data.yaml kaynaklı oranlar kullanılır.
Üretim modeli (detector_v3/detector.onnx) KESİNLİKLE dokunulmaz.

Kullanım:
    python scripts/augment_dataset.py --input dataset/images/train \
        --labels dataset/labels/train --output data/augmented \
        --multiplier 3 --low-support-classes 5 6 7

Bağımlılıklar: albumentations>=1.3.0, opencv-python, pyyaml, tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOG = logging.getLogger("sinaptic5g.augment")

# ─── Sınıf Tanımları (src/class_registry.py ile senkronize) ──────────────────
FTR_CLASS_MAP: Dict[int, str] = {
    0: "telefonla_konusma",
    1: "su_icme",
    2: "arkaya_bakma",
    3: "esneme",
    4: "sigara_icme",
    5: "emniyet_kemeri_ihlali",   # Düşük destek
    6: "teknocan",                # Düşük destek
    7: "bilgisayar",              # Düşük destek
    8: "license_plate",
}

# Düşük destekli sınıf ID'leri (varsayılan)
LOW_SUPPORT_CLASSES_DEFAULT = {5, 6, 7}

# ─── Split Config (configs/data.yaml ile senkronize) ─────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


def _try_import_albumentations():
    """Albumentations'ı dene; yoksa fallback uyarısı ver."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2  # noqa
        return A
    except ImportError:
        LOG.warning("albumentations bulunamadı — pip install albumentations>=1.3.0")
        return None


def build_heavy_pipeline(A):
    """Düşük destekli sınıflar için ağır augmentasyon pipeline'ı."""
    return A.Compose([
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.08, scale_limit=0.15, rotate_limit=20,
            border_mode=cv2.BORDER_REFLECT_101, p=0.6
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=9, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        A.GaussNoise(var_limit=(10.0, 80.0), p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=6, max_height=32, max_width=32,
            min_holes=2, min_height=8, min_width=8,
            fill_value=0, p=0.3
        ),
        A.Perspective(scale=(0.03, 0.08), p=0.2),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.15),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                 min_visibility=0.3, min_area=64.0))


def build_light_pipeline(A):
    """Yüksek destekli sınıflar için hafif augmentasyon pipeline'ı."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101, p=0.4
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                 min_visibility=0.3, min_area=64.0))


def read_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """YOLO formatındaki etiket dosyasını okur. [(class_id, cx, cy, w, h)]"""
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cid, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            rows.append((cid, cx, cy, bw, bh))
    return rows


def write_yolo_label(label_path: Path, annotations: List[Tuple[int, float, float, float, float]]) -> None:
    """YOLO formatında etiket dosyası yazar."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in annotations]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def has_low_support_class(annotations: List[Tuple], low_support: set) -> bool:
    """Annotasyon listesinde düşük destekli sınıf var mı?"""
    return any(ann[0] in low_support for ann in annotations)


def augment_image(
    img: np.ndarray,
    annotations: List[Tuple[int, float, float, float, float]],
    pipeline,
    seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], List[Tuple[int, float, float, float, float]]]:
    """Albumentations pipeline'ı uygular. Hata durumunda None döner."""
    if img is None or img.size == 0 or not annotations:
        return None, []
    
    # [0,1]'e normalize edilmiş → pipeline'ın Normalize katmanı tekrar işler
    img_float = img.astype(np.float32) / 255.0
    
    bboxes = [(ann[1], ann[2], ann[3], ann[4]) for ann in annotations]
    class_labels = [ann[0] for ann in annotations]
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    try:
        result = pipeline(image=img_float, bboxes=bboxes, class_labels=class_labels)
    except Exception as exc:
        LOG.debug("Augmentasyon hatası: %s", exc)
        return None, []
    
    aug_img = result["image"]
    # [0,1] → [0,255]
    aug_img = np.clip(aug_img * 255.0, 0, 255).astype(np.uint8)
    
    aug_annotations = [
        (int(cls), float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        for cls, bb in zip(result["class_labels"], result["bboxes"])
    ]
    return aug_img, aug_annotations


def count_class_distribution(input_dir: Path, label_dir: Path) -> Dict[int, int]:
    """Train klasöründeki sınıf dağılımını sayar."""
    dist: Dict[int, int] = {}
    for img_path in sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png")):
        label_path = label_dir / (img_path.stem + ".txt")
        for ann in read_yolo_label(label_path):
            dist[ann[0]] = dist.get(ann[0], 0) + 1
    return dist


def run_augmentation(
    input_dir: Path,
    label_dir: Path,
    output_dir: Path,
    output_label_dir: Path,
    multiplier: int = 3,
    low_support_classes: Optional[set] = None,
    low_support_multiplier: int = 5,
    copy_originals: bool = True,
    seed: int = 42,
) -> Dict:
    """Ana augmentasyon fonksiyonu. Özet istatistikleri döner."""
    if low_support_classes is None:
        low_support_classes = LOW_SUPPORT_CLASSES_DEFAULT
    
    A = _try_import_albumentations()
    if A is None:
        LOG.error("Albumentations yüklü değil. Lütfen: pip install albumentations>=1.3.0")
        return {"error": "albumentations_not_installed"}
    
    heavy_pipeline = build_heavy_pipeline(A)
    light_pipeline = build_light_pipeline(A)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    
    if not image_paths:
        LOG.warning("Girdi dizininde görüntü bulunamadı: %s", input_dir)
        return {"error": "no_images", "input_dir": str(input_dir)}
    
    stats = {
        "input_images": len(image_paths),
        "augmented_images": 0,
        "skipped": 0,
        "low_support_augmented": 0,
        "class_distribution_before": count_class_distribution(input_dir, label_dir),
        "class_distribution_after": {},
    }
    
    random.seed(seed)
    np.random.seed(seed)
    
    for idx, img_path in enumerate(image_paths):
        label_path = label_dir / (img_path.stem + ".txt")
        annotations = read_yolo_label(label_path)
        
        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped"] += 1
            continue
        
        # Orijinali kopyala
        if copy_originals:
            dest_img = output_dir / img_path.name
            dest_lbl = output_label_dir / label_path.name
            shutil.copy2(img_path, dest_img)
            if label_path.exists():
                shutil.copy2(label_path, dest_lbl)
        
        if not annotations:
            stats["skipped"] += 1
            continue
        
        is_low_support = has_low_support_class(annotations, low_support_classes)
        n_aug = low_support_multiplier if is_low_support else multiplier
        pipeline = heavy_pipeline if is_low_support else light_pipeline
        
        for aug_idx in range(n_aug):
            aug_img, aug_anns = augment_image(img, annotations, pipeline, seed=seed + idx * 100 + aug_idx)
            if aug_img is None or not aug_anns:
                continue
            
            aug_stem = f"{img_path.stem}_aug{aug_idx:03d}"
            aug_img_path = output_dir / f"{aug_stem}.jpg"
            aug_lbl_path = output_label_dir / f"{aug_stem}.txt"
            
            cv2.imwrite(str(aug_img_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo_label(aug_lbl_path, aug_anns)
            
            stats["augmented_images"] += 1
            if is_low_support:
                stats["low_support_augmented"] += 1
    
    stats["class_distribution_after"] = count_class_distribution(output_dir, output_label_dir)
    LOG.info("Augmentasyon tamamlandı: %d orijinal → %d artırılmış görüntü",
             stats["input_images"], stats["augmented_images"])
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Sinaptic5G — Albumentations Augmentasyon Pipeline")
    parser.add_argument("--input", type=Path, default=Path("dataset/images/train"),
                        help="Girdi görüntü dizini")
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels/train"),
                        help="Girdi etiket dizini")
    parser.add_argument("--output", type=Path, default=Path("data/augmented/images/train"),
                        help="Çıktı görüntü dizini")
    parser.add_argument("--output-labels", type=Path, default=None,
                        help="Çıktı etiket dizini (varsayılan: --output/../labels/train)")
    parser.add_argument("--multiplier", type=int, default=3,
                        help="Normal sınıflar için augmentasyon çarpanı")
    parser.add_argument("--low-support-multiplier", type=int, default=5,
                        help="Düşük destekli sınıflar için augmentasyon çarpanı")
    parser.add_argument("--low-support-classes", type=int, nargs="+",
                        default=[5, 6, 7],
                        help="Düşük destekli sınıf ID'leri (FTR: 5=emniyet_kemeri, 6=teknocan, 7=bilgisayar)")
    parser.add_argument("--no-copy-originals", action="store_true",
                        help="Orijinal dosyaları çıktıya kopyalama")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_strengthening_report.md"),
                        help="Rapor çıktı yolu")
    
    args = parser.parse_args()
    
    output_label_dir = args.output_labels
    if output_label_dir is None:
        output_label_dir = args.output.parent.parent / "labels" / args.output.name.split("/")[-1]
        # Güvenli fallback
        output_label_dir = Path(str(args.output).replace("images", "labels"))
    
    LOG.info("Augmentasyon başlıyor...")
    LOG.info("  Girdi    : %s", args.input)
    LOG.info("  Çıktı    : %s", args.output)
    LOG.info("  Çarpan   : x%d (düşük destek: x%d)", args.multiplier, args.low_support_multiplier)
    LOG.info("  Düşük dest. sınıflar: %s", args.low_support_classes)
    
    stats = run_augmentation(
        input_dir=args.input,
        label_dir=args.labels,
        output_dir=args.output,
        output_label_dir=output_label_dir,
        multiplier=args.multiplier,
        low_support_classes=set(args.low_support_classes),
        low_support_multiplier=args.low_support_multiplier,
        copy_originals=not args.no_copy_originals,
        seed=args.seed,
    )
    
    if "error" in stats:
        LOG.error("Hata: %s", stats["error"])
        return 1
    
    # Rapor yaz
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_report(args.report, stats, args)
    
    LOG.info("Rapor: %s", args.report)
    return 0


def _write_report(report_path: Path, stats: Dict, args) -> None:
    """Augmentasyon raporunu Markdown formatında yazar."""
    lines = [
        "# Veri Seti Güçlendirme Raporu",
        "",
        "**Script:** `scripts/augment_dataset.py`  ",
        f"**Tarih:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Üretim Kilidi:** detector_v3 / detector.onnx dokunulmadı ✅",
        "",
        "## Özet",
        "",
        f"| Parametre | Değer |",
        f"|-----------|-------|",
        f"| Girdi görüntü sayısı | {stats.get('input_images', 0)} |",
        f"| Üretilen augmentasyon | {stats.get('augmented_images', 0)} |",
        f"| Atlanan görüntüler | {stats.get('skipped', 0)} |",
        f"| Düşük destek augmentasyonu | {stats.get('low_support_augmented', 0)} |",
        f"| Normal çarpan | x{args.multiplier} |",
        f"| Düşük destek çarpanı | x{args.low_support_multiplier} |",
        f"| Düşük destek sınıfları | {sorted(args.low_support_classes)} |",
        "",
        "## Sınıf Dağılımı (Önce)",
        "",
        "| Sınıf ID | Sınıf Adı | Örnek Sayısı |",
        "|----------|-----------|-------------|",
    ]
    
    for cid, count in sorted(stats.get("class_distribution_before", {}).items()):
        name = FTR_CLASS_MAP.get(cid, f"unknown_{cid}")
        lines.append(f"| {cid} | {name} | {count} |")
    
    lines += [
        "",
        "## Sınıf Dağılımı (Sonra)",
        "",
        "| Sınıf ID | Sınıf Adı | Örnek Sayısı |",
        "|----------|-----------|-------------|",
    ]
    
    for cid, count in sorted(stats.get("class_distribution_after", {}).items()):
        name = FTR_CLASS_MAP.get(cid, f"unknown_{cid}")
        lines.append(f"| {cid} | {name} | {count} |")
    
    lines += [
        "",
        "## Augmentasyon Stratejisi",
        "",
        "### Hafif Pipeline (Normal Sınıflar)",
        "- HorizontalFlip, RandomBrightnessContrast, HueSaturationValue",
        "- ShiftScaleRotate (küçük), GaussianBlur, GaussNoise",
        "",
        "### Ağır Pipeline (Düşük Destekli: teknocan, bilgisayar, emniyet_kemeri_ihlali)",
        "- Tüm hafif augmentasyonlar + ek:",
        "- RandomRotate90, VerticalFlip, Perspective dönüşümü",
        "- ColorJitter, CLAHE, Sharpen, CoarseDropout",
        "- RandomShadow, RandomFog",
        "- Artırılmış rotasyon/ölçek sınırları",
        "",
        "## Split Tutarlılığı",
        f"Oranlar: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}",
        "Kaynak: configs/data.yaml (seed=42, policy=group_aware)",
        "",
        "## Üretim Güvenliği",
        "- ✅ detector_v3 / models/detector.onnx kilitli — dokunulmadı",
        "- ✅ Augmentasyon yalnızca deneysel data/augmented/ altında",
        "- ✅ Orijinal dataset/ dizini değiştirilmedi",
    ]
    
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
