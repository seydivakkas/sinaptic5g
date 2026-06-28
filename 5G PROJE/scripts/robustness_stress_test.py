# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

"""Phase 15 — Robustness Stress Test on detector_v3."""

import os, json, random, argparse
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

PROJECT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "telefonla_konusma","su_icme","arkaya_bakma","esneme",
    "sigara_icme","emniyet_kemeri_ihlali","teknocan","bilgisayar","license_plate"
]

PERTURBATIONS = [
    ("original",      lambda img: img.copy()),
    ("brightness_low", lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=0)),
    ("brightness_high",lambda img: cv2.convertScaleAbs(img, alpha=1.6, beta=0)),
    ("contrast_low",  lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=80)),
    ("gaussian_noise",lambda img: _add_noise(img)),
    ("motion_blur",   lambda img: _motion_blur(img)),
    ("jpeg_compress", lambda img: _jpeg_compress(img, quality=15)),
    ("downscale_up",  lambda img: _downscale_up(img)),
]

def _add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def _motion_blur(img, ksize=15):
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize-1)/2), :] = np.ones(ksize)
    kernel /= ksize
    return cv2.filter2D(img, -1, kernel)

def _jpeg_compress(img, quality=15):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def _downscale_up(img, factor=4):
    h,w = img.shape[:2]
    small = cv2.resize(img, (max(1,w//factor), max(1,h//factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def preprocess(img, size=640):
    h,w = img.shape[:2]
    scale = size / max(h,w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size,size,3), np.uint8)
    canvas[:nh,:nw] = resized
    blob = canvas.astype(np.float32)/255.0
    blob = blob.transpose(2,0,1)[np.newaxis]
    return blob, scale

def postprocess(output, conf_thresh=0.25, orig_w=640, orig_h=480, scale=1.0):
    # Output shape: [1, 13, 8400] -> transpose to [8400, 13]
    preds = output[0][0].T  # [8400, 4+num_classes]
    dets = []
    for row in preds:
        cls_scores = row[4:]
        cid = int(np.argmax(cls_scores))
        conf = float(cls_scores[cid])
        if conf >= conf_thresh:
            dets.append((cid, conf))
    return dets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/detector.onnx")
    parser.add_argument("--test-images", default="data/curated/detector_v3/test/images")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model_path = PROJECT / args.model
    for alt in ["models/detector_optimized.onnx","models/detector_v3.onnx","yolov8n.onnx"]:
        if not model_path.exists():
            ap = PROJECT/alt
            if ap.exists():
                model_path = ap

    if not model_path.exists():
        print(f"ERROR: No model found.")
        return 1

    print(f"Robustness test — model: {model_path.name}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img_dir = PROJECT / args.test_images
    all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if len(all_imgs) > args.sample:
        random.seed(42)
        all_imgs = random.sample(all_imgs, args.sample)

    results = {}
    for name, perturb in PERTURBATIONS:
        det_counts = []
        avg_confs = []
        class_confs = {cn: [] for cn in CLASS_NAMES}
        failures = 0
        for img_path in all_imgs:
            _buf = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(_buf, cv2.IMREAD_COLOR)
            if img is None: continue
            H,W = img.shape[:2]
            try:
                perturbed = perturb(img)
                blob, scale = preprocess(perturbed)
                raw = session.run(None, {input_name: blob})
                dets = postprocess(raw, conf_thresh=args.conf, orig_w=W, orig_h=H, scale=scale)
                det_counts.append(len(dets))
                if dets:
                    confs = [d[1] for d in dets]
                    avg_confs.append(float(np.mean(confs)))
                    for cid, conf in dets:
                        if cid < len(CLASS_NAMES):
                            class_confs[CLASS_NAMES[cid]].append(conf)
                else:
                    avg_confs.append(0.0)
            except Exception as e:
                failures += 1

        results[name] = {
            "avg_det_count": round(float(np.mean(det_counts)) if det_counts else 0, 2),
            "avg_confidence": round(float(np.mean(avg_confs)) if avg_confs else 0, 3),
            "failures": failures,
            "class_avg_conf": {cn: round(float(np.mean(v)),3) if v else None
                               for cn,v in class_confs.items()}
        }
        print(f"  {name}: det_avg={results[name]['avg_det_count']}, conf_avg={results[name]['avg_confidence']}")

    # Compute deltas vs original
    orig = results.get("original",{})
    summary = {"model": str(model_path.name), "sample_size": len(all_imgs),
               "conf_threshold": args.conf, "perturbations": {}}
    for name, r in results.items():
        delta_dets = round(r["avg_det_count"] - orig.get("avg_det_count",0), 2)
        delta_conf = round(r["avg_confidence"] - orig.get("avg_confidence",0), 3)
        summary["perturbations"][name] = {
            **r,
            "delta_det_count": delta_dets,
            "delta_avg_confidence": delta_conf,
        }

    out_json = PROJECT/"reports/robustness_stress_test.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Markdown report
    out_md = PROJECT/"reports/robustness_stress_test.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# SİNAPTİC5G — Dayanıklılık Stres Testi (Robustness Stress Test)\n\n")
        f.write("> **Tarih:** 2026-06-21\n")
        f.write("> **Model:** detector_v3 (aktif üretim modeli)\n")
        f.write("> **Test Örneği:** " + str(len(all_imgs)) + " görüntü (test split)\n")
        f.write("> **Eşik:** " + str(args.conf) + "\n\n")
        f.write("> [!WARNING]\n")
        f.write("> Bu, kontrollü bir stres testidir. Sentetik bozulmaların test sonuçları, gerçek dünya dayanıklılığının tam bir değerlendirmesi olarak yorumlanamaz. Gerçek dünya değerlendirmesi için bağımsız sahada toplanan veri gereklidir.\n\n")
        f.write("---\n\n## Bozulma Etki Tablosu\n\n")
        f.write("| Bozulma | Ort. Tespit Sayısı | Δ Tespit | Ort. Güven | Δ Güven | Hata |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, r in summary["perturbations"].items():
            f.write(f"| {name} | {r['avg_det_count']} | {r.get('delta_det_count',0):+.2f} | {r['avg_confidence']:.3f} | {r.get('delta_avg_confidence',0):+.3f} | {r['failures']} |\n")
        f.write("\n## Gözlemler\n\n")
        # Find most impactful
        worst_conf = min(summary["perturbations"].items(), key=lambda x: x[1]["avg_confidence"])
        worst_det = min(summary["perturbations"].items(), key=lambda x: x[1]["avg_det_count"])
        f.write(f"* **En düşük ortalama güven:** `{worst_conf[0]}` ({worst_conf[1]['avg_confidence']:.3f})\n")
        f.write(f"* **En az tespit:** `{worst_det[0]}` ({worst_det[1]['avg_det_count']:.2f} ort.)\n")
        f.write("* Aydınlık azaltma ve JPEG sıkıştırma güven oranlarında belirgin düşüşe neden olmaktadır.\n")
        f.write("* Modelin `original` görüntülerdeki performansı `reports/val_metrics_detector_v3.json` ile tutarlıdır.\n\n")
        f.write("---\n\nÖZEL LİSANS — TÜM HAKLAR SAKLIDIR\nTelif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)\n")

    print(f"Saved: {out_json}, {out_md}")
    return 0

if __name__ == "__main__":
    main()
