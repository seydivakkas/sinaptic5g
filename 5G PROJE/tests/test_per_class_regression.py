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
tests/test_per_class_regression.py — Sınıf Bazlı Regresyon Testi
=================================================================
Faz 5: Test & Doğrulama

Üretim modeli (models/detector.onnx = detector_v5) üzerinde sınıf bazlı
regresyon testi yapar. Metrikler beklenen minimumların altına düşerse FAIL.

Üretim kilidi: Sadece detector_v5 / models/detector.onnx kullanılır.
Bu test modeli ASLA değiştirmez.

Çalıştırma:
    pytest tests/test_per_class_regression.py -v
    # veya
    python tests/test_per_class_regression.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("sinaptic5g.test_regression")

# ─── Üretim Modeli Sabitleri ──────────────────────────────────────────────────
PRODUCTION_MODEL_PATH = PROJECT_ROOT / "models" / "detector.onnx"
MODEL_SIZE = 640

# FTR Sınıf Haritası (src/class_registry.py ile senkronize)
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

# ─── Beklenen Minimum Metrikler (Detector_v5 taban çizgisi) ──────────────────
# Kaynak: reports/detector_v5_vs_v3_comparison.md
# Regresyon: Bu değerlerin ALTINA DÜŞMESİ test başarısızlığını tetikler
EXPECTED_MIN_METRICS: Dict[str, Dict[str, float]] = {
    "telefonla_konusma":      {"precision": 0.50, "recall": 0.10, "f1": 0.10},
    "su_icme":                {"precision": 0.70, "recall": 0.80, "f1": 0.75},
    "arkaya_bakma":           {"precision": 0.70, "recall": 0.80, "f1": 0.75},
    "esneme":                 {"precision": 0.30, "recall": 0.25, "f1": 0.25},
    "sigara_icme":            {"precision": 0.70, "recall": 0.75, "f1": 0.70},
    "emniyet_kemeri_ihlali":  {"precision": 0.20, "recall": 0.05, "f1": 0.04},
    "teknocan":               {"precision": 0.00, "recall": 0.00, "f1": 0.00},
    "bilgisayar":             {"precision": 0.00, "recall": 0.00, "f1": 0.00},
    "license_plate":          {"precision": 0.70, "recall": 0.75, "f1": 0.70},
}

# Genel mAP50 minimum eşiği
MIN_OVERALL_MAP50 = 0.45


# ─── ONNX Model Yükleyici (Salt Okunur) ──────────────────────────────────────

def load_production_model():
    """Üretim modelini salt okunur olarak yükler."""
    if not PRODUCTION_MODEL_PATH.is_file():
        pytest.skip(f"Üretim modeli bulunamadı: {PRODUCTION_MODEL_PATH}")
    
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess = ort.InferenceSession(str(PRODUCTION_MODEL_PATH), sess_opts, providers=providers)
        return sess
    except ImportError:
        pytest.skip("onnxruntime kurulu değil")
    except Exception as e:
        pytest.skip(f"Model yüklenemedi: {e}")


def predict_on_image(session, img: np.ndarray, conf_thresh: float = 0.25) -> List[Dict]:
    """Model ile tahmin yap (NMS uygulanmış şekilde)."""
    import cv2
    h, w = img.shape[:2]
    resized = cv2.resize(img, (MODEL_SIZE, MODEL_SIZE))
    blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    
    input_name = session.get_inputs()[0].name
    raw = np.asarray(session.run(None, {input_name: blob})[0])
    preds = raw[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T
    
    boxes = []
    scores = []
    class_ids = []
    
    scale_x, scale_y = w / MODEL_SIZE, h / MODEL_SIZE
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
        
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(conf)
        class_ids.append(class_id)
        
    if not boxes:
        return []
        
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, 0.45)
    results = []
    for index in np.asarray(indices).reshape(-1).tolist():
        x, y, bw, bh = boxes[index]
        results.append({
            "class_id": class_ids[index],
            "conf": scores[index],
            "bbox": [x, y, x + bw, y + bh]
        })
    return results


def compute_iou(b1, b2) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-6)


def evaluate_on_test_set(
    session,
    test_images_dir: Path,
    test_labels_dir: Path,
    iou_threshold: float = 0.5,
    conf_thresh: float = 0.25,
) -> Dict:
    """Test seti üzerinde per-class metrik hesaplar (stratified sampling ile)."""
    import cv2
    import random
    
    label_paths = list(test_labels_dir.glob("*.txt"))
    class_to_paths = {cid: [] for cid in FTR_CLASS_MAP}
    
    for lbl_path in label_paths:
        img_path = test_images_dir / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            img_path = test_images_dir / (lbl_path.stem + ".png")
        if not img_path.exists():
            continue
            
        try:
            content = lbl_path.read_text(encoding="utf-8").strip()
            for line in content.splitlines():
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    if cid in class_to_paths:
                        class_to_paths[cid].append(img_path)
        except Exception:
            continue
            
    # Deterministic stratified sample (up to 15 per class)
    selected_images = set()
    rng = random.Random(42)
    for cid, paths in class_to_paths.items():
        if len(paths) > 15:
            selected_images.update(rng.sample(paths, 15))
        else:
            selected_images.update(paths)
            
    image_paths = sorted(list(selected_images))
    
    if not image_paths:
        return {"error": "no_test_images", "evaluated": 0}
    
    per_class_tp: Dict[int, int] = {cid: 0 for cid in FTR_CLASS_MAP}
    per_class_fp: Dict[int, int] = {cid: 0 for cid in FTR_CLASS_MAP}
    per_class_fn: Dict[int, int] = {cid: 0 for cid in FTR_CLASS_MAP}
    
    for img_path in image_paths:
        label_path = test_labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Ground truth
        gt_boxes: List[Dict] = []
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cid = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                gt_boxes.append({"class_id": cid, "bbox": [x1, y1, x2, y2], "matched": False})
        
        # Tahminler
        predictions = predict_on_image(session, img, conf_thresh)
        
        # Eşleştirme
        for pred in predictions:
            pred_cid = pred["class_id"]
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt["class_id"] != pred_cid or gt["matched"]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                per_class_tp[pred_cid] += 1
                gt_boxes[best_gt_idx]["matched"] = True
            elif pred_cid in per_class_fp:
                per_class_fp[pred_cid] += 1
        
        # False negatives
        for gt in gt_boxes:
            if not gt["matched"]:
                per_class_fn[gt["class_id"]] += 1
    
    # Metrik hesapla
    results = {"evaluated": len(image_paths), "per_class": {}}
    for cid, name in FTR_CLASS_MAP.items():
        tp = per_class_tp[cid]
        fp = per_class_fp[cid]
        fn = per_class_fn[cid]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results["per_class"][name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    
    return results


# ─── Pytest Testleri ──────────────────────────────────────────────────────────

class TestPerClassRegression:
    """Sınıf bazlı regresyon testleri — üretim modeli (detector_v5) üzerinde."""
    
    @pytest.fixture(scope="class")
    def production_session(self):
        return load_production_model()
    
    @pytest.fixture(scope="class")
    def test_dirs(self):
        img_dir = PROJECT_ROOT / "data" / "curated" / "detector_v5" / "test" / "images"
        lbl_dir = PROJECT_ROOT / "data" / "curated" / "detector_v5" / "test" / "labels"
        if not img_dir.exists():
            img_dir = PROJECT_ROOT / "dataset" / "images" / "test"
            lbl_dir = PROJECT_ROOT / "dataset" / "labels" / "test"
        if not img_dir.exists():
            pytest.skip(f"Test dizini bulunamadı: {img_dir}")
        return img_dir, lbl_dir
    
    @pytest.fixture(scope="class")
    def evaluation_results(self, production_session, test_dirs):
        img_dir, lbl_dir = test_dirs
        results = evaluate_on_test_set(production_session, img_dir, lbl_dir)
        # Sonuçları dosyaya yaz
        report_path = PROJECT_ROOT / "reports" / "per_class_regression_test_results.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        return results
    
    def test_production_model_exists(self):
        """Üretim modeli mevcut olmalı."""
        assert PRODUCTION_MODEL_PATH.is_file(), \
            f"Üretim modeli bulunamadı: {PRODUCTION_MODEL_PATH}"
    
    def test_production_model_lock_file_exists(self):
        """Model kilidi dosyası mevcut olmalı."""
        lock_path = PROJECT_ROOT / "model_lock.json"
        assert lock_path.is_file(), f"model_lock.json bulunamadı: {lock_path}"
    
    def test_model_loaded_successfully(self, production_session):
        """Model başarıyla yüklenmeli."""
        assert production_session is not None
    
    def test_model_produces_output(self, production_session):
        """Model dummy input üzerinde çıktı üretmeli."""
        dummy = np.zeros((1, 3, MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
        input_name = production_session.get_inputs()[0].name
        outputs = production_session.run(None, {input_name: dummy})
        assert outputs is not None and len(outputs) > 0
    
    @pytest.mark.parametrize("class_name,min_metrics", [
        (name, metrics) for name, metrics in EXPECTED_MIN_METRICS.items()
    ])
    def test_per_class_recall(self, evaluation_results, class_name, min_metrics):
        """Sınıf bazlı recall beklenen minimumun üzerinde olmalı."""
        if "error" in evaluation_results:
            pytest.skip(f"Değerlendirme hatası: {evaluation_results['error']}")
        
        per_class = evaluation_results.get("per_class", {})
        if class_name not in per_class:
            pytest.skip(f"Sınıf test setinde mevcut değil: {class_name}")
        
        actual_recall = per_class[class_name]["recall"]
        expected_recall = min_metrics["recall"]
        
        assert actual_recall >= expected_recall, (
            f"RECALL REGRESYONU [{class_name}]: "
            f"beklenen >= {expected_recall:.3f}, gerçek = {actual_recall:.3f}"
        )
    
    @pytest.mark.parametrize("class_name,min_metrics", [
        (name, metrics) for name, metrics in EXPECTED_MIN_METRICS.items()
    ])
    def test_per_class_precision(self, evaluation_results, class_name, min_metrics):
        """Sınıf bazlı precision beklenen minimumun üzerinde olmalı."""
        if "error" in evaluation_results:
            pytest.skip(f"Değerlendirme hatası: {evaluation_results['error']}")
        
        per_class = evaluation_results.get("per_class", {})
        if class_name not in per_class:
            pytest.skip(f"Sınıf test setinde mevcut değil: {class_name}")
        
        actual_precision = per_class[class_name]["precision"]
        expected_precision = min_metrics["precision"]
        
        assert actual_precision >= expected_precision, (
            f"PRECISION REGRESYONU [{class_name}]: "
            f"beklenen >= {expected_precision:.3f}, gerçek = {actual_precision:.3f}"
        )
    
    def test_no_model_file_mutation(self):
        """Üretim modeli test süresince değiştirilmemeli."""
        import hashlib
        lock_path = PROJECT_ROOT / "model_lock.json"
        if not lock_path.is_file() or not PRODUCTION_MODEL_PATH.is_file():
            pytest.skip("Model veya lock dosyası yok")
        
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        expected_sha = lock.get("detector_onnx_sha256", "")
        
        if not expected_sha:
            pytest.skip("detector_onnx_sha256 lock'ta bulunamadı")
        
        digest = hashlib.sha256()
        with PRODUCTION_MODEL_PATH.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        
        assert actual_sha.lower() == expected_sha.lower(), (
            "ÜRETIM MODELİ BÜTÜNLÜK HATASI: model_lock.json SHA256 eşleşmiyor!\n"
            f"Beklenen: {expected_sha}\n"
            f"Gerçek  : {actual_sha}"
        )


# ─── Standalone çalıştırma ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    LOG.info("Per-class regresyon testi başlıyor...")
    LOG.info("  Üretim modeli: %s", PRODUCTION_MODEL_PATH)
    
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(PRODUCTION_MODEL_PATH), providers=["CPUExecutionProvider"])
        LOG.info("  Model yüklendi ✅")
    except Exception as e:
        LOG.error("Model yüklenemedi: %s", e)
        sys.exit(1)
    
    img_dir = PROJECT_ROOT / "dataset" / "images" / "test"
    lbl_dir = PROJECT_ROOT / "dataset" / "labels" / "test"
    
    if not img_dir.exists():
        LOG.warning("Test dizini bulunamadı: %s — smoke test modu", img_dir)
        LOG.info("Test atlandı — pytest ile çalıştırın: pytest tests/test_per_class_regression.py -v")
        sys.exit(0)
    
    results = evaluate_on_test_set(sess, img_dir, lbl_dir)
    
    LOG.info("\n=== Per-Class Regresyon Sonuçları ===")
    all_passed = True
    for class_name, metrics in results.get("per_class", {}).items():
        expected = EXPECTED_MIN_METRICS.get(class_name, {})
        status = "✅"
        issues = []
        if metrics["recall"] < expected.get("recall", 0):
            issues.append(f"recall={metrics['recall']:.3f} < {expected['recall']:.3f}")
            status = "❌"
        if metrics["precision"] < expected.get("precision", 0):
            issues.append(f"precision={metrics['precision']:.3f} < {expected['precision']:.3f}")
            status = "❌"
        if status == "❌":
            all_passed = False
        LOG.info("  %s [%s] P=%.3f R=%.3f F1=%.3f %s",
                  status, class_name, metrics["precision"], metrics["recall"], metrics["f1"],
                  f"— {', '.join(issues)}" if issues else "")
    
    LOG.info("\nSonuç: %s", "✅ TÜM SINIFLAR PASS" if all_passed else "❌ BAZI SINIFLAR FAIL")
    sys.exit(0 if all_passed else 1)
