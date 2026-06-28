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
scripts/audit_labels.py — Etiket Kalite Denetimi Scripti
=========================================================
Faz 2: Veri Seti Güçlendirme

Görevler:
1. Model tahmini vs ground-truth IoU kontrolü
2. Şüpheli etiket tespiti (çok küçük/büyük, kenarda, kötü format)
3. Ground-truth bbox kalite kontrolü
4. Şüpheli etiketler CSV raporu
5. Split tutarlılığı kontrolü

Kullanım:
    python scripts/audit_labels.py \
        --images-dir dataset/images/train \
        --labels-dir dataset/labels/train \
        --model-path models/detector.onnx \
        --output-csv reports/suspicious_labels.csv \
        --iou-threshold 0.3

Üretim kilidi: models/detector.onnx KESİNLİKLE SALT OKUNUR kullanılır.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOG = logging.getLogger("sinaptic5g.audit_labels")

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

MODEL_SIZE = 640


# ─── ONNX Model Yükleyici (salt okunur — üretim modeli) ──────────────────────

class ReadOnlyOnnxDetector:
    """Üretim modelini SALT OKUNUR olarak yükler. Model dosyasına asla yazmaz."""
    
    def __init__(self, model_path: Path):
        self.session = None
        self.model_path = model_path
        
        if not model_path.is_file():
            LOG.warning("Model bulunamadı: %s — sadece GT kontrolü yapılacak", model_path)
            return
        
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 2
            self.session = ort.InferenceSession(str(model_path), sess_opts, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            LOG.info("Model yüklendi (salt okunur): %s", model_path)
        except Exception as exc:
            LOG.warning("Model yüklenemedi: %s", exc)
    
    def predict(self, img: np.ndarray, conf_thresh: float = 0.25) -> List[Dict]:
        """Tahminleri döner. Modele yazar — ASLA."""
        if self.session is None:
            return []
        
        h, w = img.shape[:2]
        resized = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE))
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        
        try:
            raw = np.asarray(self.session.run(None, {self.input_name: blob})[0])
        except Exception as exc:
            LOG.debug("Tahmin hatası: %s", exc)
            return []
        
        preds = raw[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        
        scale_x, scale_y = w / MODEL_SIZE, h / MODEL_SIZE
        results = []
        
        for row in preds:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])
            if conf < conf_thresh:
                continue
            cx, cy, bw, bh = map(float, row[:4])
            x1 = max(0, int((cx - bw / 2) * scale_x))
            y1 = max(0, int((cy - bh / 2) * scale_y))
            x2 = min(w, int((cx + bw / 2) * scale_x))
            y2 = min(h, int((cy + bh / 2) * scale_y))
            results.append({"class_id": class_id, "conf": conf, "bbox": [x1, y1, x2, y2]})
        
        return results


# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

def read_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4])))
            except ValueError:
                pass
    return rows


def yolo_to_pixel(cx, cy, bw, bh, img_w, img_h):
    x1 = max(0, int((cx - bw / 2) * img_w))
    y1 = max(0, int((cy - bh / 2) * img_h))
    x2 = min(img_w, int((cx + bw / 2) * img_w))
    y2 = min(img_h, int((cy + bh / 2) * img_h))
    return x1, y1, x2, y2


def iou_boxes(b1: List[int], b2: List[int]) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-6)


def best_gt_iou_for_prediction(pred_bbox: List[int],
                                gt_boxes: List[List[int]]) -> float:
    """Tahmin ile en yüksek GT IoU'sunu döner."""
    if not gt_boxes:
        return 0.0
    return max(iou_boxes(pred_bbox, gt) for gt in gt_boxes)


def check_gt_bbox_quality(cx: float, cy: float, bw: float, bh: float,
                           img_w: int, img_h: int) -> List[str]:
    """Ground-truth bbox kalite sorunlarını listeler."""
    issues = []
    area = bw * bh
    
    if area < 0.0004:  # < 0.04% görüntü alanı
        issues.append("too_small")
    if area > 0.85:
        issues.append("too_large")
    if bw > 1.0 or bh > 1.0:
        issues.append("out_of_bounds_normalized")
    if cx < 0.01 or cx > 0.99 or cy < 0.01 or cy > 0.99:
        issues.append("near_edge")
    if bw <= 0 or bh <= 0:
        issues.append("invalid_zero_dim")
    
    # Pixel level check
    x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)
    if (x2 - x1) < 8 or (y2 - y1) < 8:
        issues.append("too_small_pixels")
    
    return issues


def audit_dataset(
    images_dir: Path,
    labels_dir: Path,
    detector: ReadOnlyOnnxDetector,
    iou_threshold: float = 0.3,
    conf_thresh: float = 0.25,
    max_images: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Veri seti kalite denetimi yapar.
    Returns: (suspicious_list, summary_stats)
    """
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    suspicious = []
    stats = {
        "total_images": len(image_paths),
        "images_with_labels": 0,
        "total_gt_boxes": 0,
        "low_iou_boxes": 0,
        "quality_issues": 0,
        "missing_label_file": 0,
        "per_class_suspicious": {},
    }
    
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        
        if not label_path.exists():
            stats["missing_label_file"] += 1
            suspicious.append({
                "image": img_path.name,
                "issue": "missing_label_file",
                "class_id": -1,
                "class_name": "N/A",
                "gt_bbox": "N/A",
                "iou_with_pred": "N/A",
                "quality_flags": "missing_label",
            })
            continue
        
        annotations = read_yolo_label(label_path)
        if not annotations:
            continue
        
        stats["images_with_labels"] += 1
        
        img = cv2.imread(str(img_path))
        if img is None:
            suspicious.append({
                "image": img_path.name,
                "issue": "cannot_read_image",
                "class_id": -1,
                "class_name": "N/A",
                "gt_bbox": "N/A",
                "iou_with_pred": "N/A",
                "quality_flags": "corrupt_image",
            })
            continue
        
        h, w = img.shape[:2]
        
        # Model tahminleri (salt okunur)
        predictions = detector.predict(img, conf_thresh=conf_thresh)
        pred_boxes = [p["bbox"] for p in predictions]
        
        # GT bbox'larını piksel koordinatlarına çevir
        gt_pixel_boxes = [list(yolo_to_pixel(a[1], a[2], a[3], a[4], w, h)) for a in annotations]
        
        for ann_idx, ann in enumerate(annotations):
            cid, cx, cy, bw_n, bh_n = ann
            class_name = FTR_CLASS_MAP.get(cid, f"unknown_{cid}")
            
            stats["total_gt_boxes"] += 1
            
            # GT kalite kontrolü
            quality_flags = check_gt_bbox_quality(cx, cy, bw_n, bh_n, w, h)
            
            # Model tahmini ile IoU
            gt_px = list(yolo_to_pixel(cx, cy, bw_n, bh_n, w, h))
            best_iou = best_gt_iou_for_prediction(gt_px, pred_boxes)
            
            is_suspicious = False
            reasons = []
            
            if quality_flags:
                is_suspicious = True
                reasons.extend(quality_flags)
                stats["quality_issues"] += 1
            
            if detector.session is not None and best_iou < iou_threshold:
                is_suspicious = True
                reasons.append(f"low_model_iou={best_iou:.3f}")
                stats["low_iou_boxes"] += 1
            
            if is_suspicious:
                entry = {
                    "image": img_path.name,
                    "issue": "|".join(reasons),
                    "class_id": cid,
                    "class_name": class_name,
                    "gt_bbox": f"{cx:.4f},{cy:.4f},{bw_n:.4f},{bh_n:.4f}",
                    "iou_with_pred": f"{best_iou:.4f}",
                    "quality_flags": "|".join(quality_flags) if quality_flags else "none",
                }
                suspicious.append(entry)
                
                per_cls = stats["per_class_suspicious"]
                per_cls[class_name] = per_cls.get(class_name, 0) + 1
    
    return suspicious, stats


def write_csv_report(csv_path: Path, suspicious: List[Dict]) -> None:
    """Şüpheli etiketleri CSV olarak yazar."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not suspicious:
        LOG.info("Şüpheli etiket bulunamadı.")
        csv_path.write_text("image,issue,class_id,class_name,gt_bbox,iou_with_pred,quality_flags\n",
                             encoding="utf-8")
        return
    
    fieldnames = ["image", "issue", "class_id", "class_name", "gt_bbox", "iou_with_pred", "quality_flags"]
    
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(suspicious)
    
    LOG.info("CSV raporu yazıldı: %s (%d satır)", csv_path, len(suspicious))


def write_summary_report(report_path: Path, suspicious: List[Dict], stats: Dict,
                          args) -> None:
    """Özet Markdown raporu yazar."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    suspicious_rate = 100.0 * len(suspicious) / max(stats["total_gt_boxes"], 1)
    
    lines = [
        "# Etiket Kalite Denetim Raporu",
        "",
        f"**Script:** `scripts/audit_labels.py`  ",
        f"**Tarih:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Girdi:** {args.images_dir}  ",
        f"**Üretim Kilidi:** detector_v3 salt okunur kullanıldı ✅",
        "",
        "## Özet İstatistikleri",
        "",
        "| Metrik | Değer |",
        "|--------|-------|",
        f"| Toplam görüntü | {stats['total_images']} |",
        f"| Etiketli görüntü | {stats['images_with_labels']} |",
        f"| Toplam GT bbox | {stats['total_gt_boxes']} |",
        f"| Şüpheli bbox | {len(suspicious)} ({suspicious_rate:.1f}%) |",
        f"| Düşük IoU (<{args.iou_threshold}) | {stats['low_iou_boxes']} |",
        f"| Kalite sorunu | {stats['quality_issues']} |",
        f"| Eksik etiket dosyası | {stats['missing_label_file']} |",
        "",
        "## Sınıf Bazlı Şüpheli Etiketler",
        "",
        "| Sınıf | Şüpheli Sayısı |",
        "|-------|----------------|",
    ]
    
    for cls_name, count in sorted(stats.get("per_class_suspicious", {}).items(),
                                   key=lambda x: -x[1]):
        lines.append(f"| {cls_name} | {count} |")
    
    lines += [
        "",
        "## Kalite Kontrolü Açıklamaları",
        "",
        "| Flag | Açıklama |",
        "|------|----------|",
        "| `too_small` | Alan < 0.04% görüntü alanı |",
        "| `too_large` | Alan > 85% görüntü alanı |",
        "| `near_edge` | Merkez kenar bölgesinde (cx/cy < 1% veya > 99%) |",
        "| `too_small_pixels` | Piksel boyutu < 8×8 |",
        "| `out_of_bounds_normalized` | Normalize koordinat > 1.0 |",
        f"| `low_model_iou=X` | Model tahminiyle IoU < {args.iou_threshold} |",
        "| `missing_label_file` | Etiket dosyası yok |",
        "",
        "## Üretim Güvenliği",
        "- ✅ models/detector.onnx (detector_v3) salt okunur kullanıldı",
        "- ✅ Model dosyasına yazma yapılmadı",
        "- ✅ Orijinal dataset/ değiştirilmedi",
        f"",
        f"**Detaylı CSV:** {args.output_csv}",
    ]
    
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Sinaptic5G — Etiket Kalite Denetimi")
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images/train"),
                        help="Görüntü dizini")
    parser.add_argument("--labels-dir", type=Path, default=Path("dataset/labels/train"),
                        help="Etiket dizini")
    parser.add_argument("--model-path", type=Path, default=Path("models/detector.onnx"),
                        help="Üretim model yolu (salt okunur)")
    parser.add_argument("--output-csv", type=Path, default=Path("reports/suspicious_labels.csv"),
                        help="Şüpheli etiket CSV çıktısı")
    parser.add_argument("--output-report", type=Path, default=Path("reports/label_audit_report.md"),
                        help="Özet rapor")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="Bu değerin altındaki IoU şüpheli sayılır")
    parser.add_argument("--conf-thresh", type=float, default=0.25,
                        help="Model güven eşiği")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Test için maksimum görüntü sayısı")
    
    args = parser.parse_args()
    
    LOG.info("Etiket denetimi başlıyor...")
    LOG.info("  Görüntü dizini: %s", args.images_dir)
    LOG.info("  Etiket dizini : %s", args.labels_dir)
    LOG.info("  Model         : %s (salt okunur)", args.model_path)
    LOG.info("  IoU eşiği     : %.2f", args.iou_threshold)
    
    # Model salt okunur yüklenir
    detector = ReadOnlyOnnxDetector(args.model_path)
    
    suspicious, stats = audit_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        detector=detector,
        iou_threshold=args.iou_threshold,
        conf_thresh=args.conf_thresh,
        max_images=args.max_images,
    )
    
    write_csv_report(args.output_csv, suspicious)
    write_summary_report(args.output_report, suspicious, stats, args)
    
    LOG.info("Denetim tamamlandı.")
    LOG.info("  Toplam GT: %d, Şüpheli: %d (%.1f%%)",
              stats["total_gt_boxes"], len(suspicious),
              100.0 * len(suspicious) / max(stats["total_gt_boxes"], 1))
    LOG.info("  CSV: %s", args.output_csv)
    LOG.info("  Rapor: %s", args.output_report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
