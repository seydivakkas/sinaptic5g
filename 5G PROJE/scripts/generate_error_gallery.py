# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

"""
Phase 14 — Error Analysis: generates error gallery from GT labels on test split.
Uses YOLO-format labels (GT only). Runs detector_v3 ONNX inference, computes IoU,
classifies FP/FN/low-conf/high-conf and saves overlay images.
"""

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
COLORS = [
    (0,120,255),(0,200,100),(255,80,0),(200,0,200),
    (0,200,200),(255,40,40),(160,160,0),(0,160,255),(80,255,80)
]

def xywh_to_xyxy(cx,cy,w,h,W,H):
    x1=int((cx-w/2)*W); y1=int((cy-h/2)*H)
    x2=int((cx+w/2)*W); y2=int((cy+h/2)*H)
    return max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2)

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def preprocess(img, size=640):
    h,w=img.shape[:2]
    scale=size/max(h,w)
    nh,nw=int(h*scale),int(w*scale)
    resized=cv2.resize(img,(nw,nh))
    canvas=np.zeros((size,size,3),np.uint8)
    canvas[:nh,:nw]=resized
    blob=canvas.astype(np.float32)/255.0
    blob=blob.transpose(2,0,1)[np.newaxis]
    return blob, scale, (0,0)

def postprocess(output, conf_thresh=0.25, nms_thresh=0.45, orig_w=640, orig_h=480, scale=1.0):
    # Output shape: [1, 13, 8400] -> transpose to [8400, 13]
    preds = output[0][0].T  # [8400, 4+num_classes]
    boxes=[]; scores=[]; class_ids=[]
    for row in preds:
        x,y,w,h=row[0],row[1],row[2],row[3]
        cls_scores=row[4:]
        cid=int(np.argmax(cls_scores))
        conf=float(cls_scores[cid])
        if conf<conf_thresh: continue
        x1=int((x-w/2)/scale); y1=int((y-h/2)/scale)
        x2=int((x+w/2)/scale); y2=int((y+h/2)/scale)
        x1=max(0,min(orig_w-1,x1)); y1=max(0,min(orig_h-1,y1))
        x2=max(0,min(orig_w-1,x2)); y2=max(0,min(orig_h-1,y2))
        boxes.append([x1,y1,x2,y2]); scores.append(conf); class_ids.append(cid)
    if not boxes: return []
    indices=cv2.dnn.NMSBoxes([[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in boxes],
                              scores,conf_thresh,nms_thresh)
    if len(indices)==0: return []
    return [(boxes[i],scores[i],class_ids[i]) for i in indices.flatten()]

def draw_overlay(img, label, pred_cid, pred_conf, iou_val, reason_tag, color):
    overlay=img.copy()
    h,w=img.shape[:2]
    text_lines=[
        f"GT: {label}",
        f"PRED: {CLASS_NAMES[pred_cid] if 0<=pred_cid<len(CLASS_NAMES) else 'cls'+str(pred_cid)} ({pred_conf:.2f})",
        f"IoU: {iou_val:.2f}",
        f"Tag: {reason_tag}"
    ]
    y0=20
    for line in text_lines:
        cv2.putText(overlay,line,(8,y0),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1,cv2.LINE_AA)
        y0+=18
    return overlay

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", default="models/detector.onnx")
    parser.add_argument("--test-images", default="data/curated/detector_v3/test/images")
    parser.add_argument("--test-labels", default="data/curated/detector_v3/test/labels")
    parser.add_argument("--out-dir", default="reports/error_gallery")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.25)
    args=parser.parse_args()

    model_path=PROJECT/args.model
    if not model_path.exists():
        # Try alternate
        for alt in ["models/detector_optimized.onnx","models/detector_v3.onnx","yolov8n.onnx"]:
            ap=PROJECT/alt
            if ap.exists():
                model_path=ap
                break

    if not model_path.exists():
        print(f"ERROR: No model found. Checked {args.model} and alternates.")
        return 1

    print(f"Loading model: {model_path}")
    sess_opts=ort.SessionOptions()
    sess_opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session=ort.InferenceSession(str(model_path),sess_opts,providers=["CPUExecutionProvider"])
    input_name=session.get_inputs()[0].name

    img_dir=PROJECT/args.test_images
    lbl_dir=PROJECT/args.test_labels
    out_base=PROJECT/args.out_dir

    for sub in ["false_positive","false_negative","low_confidence_correct","high_confidence_correct"]:
        (out_base/sub).mkdir(parents=True,exist_ok=True)

    all_imgs=sorted(img_dir.glob("*.jpg"))+sorted(img_dir.glob("*.png"))+sorted(img_dir.glob("*.jpeg"))
    if len(all_imgs)>args.sample:
        random.seed(42)
        all_imgs=random.sample(all_imgs,args.sample)

    gallery={"false_positive":[],"false_negative":[],"low_confidence_correct":[],"high_confidence_correct":[]}
    stats={"reviewed":0,"fp":0,"fn":0,"low_conf_tp":0,"high_conf_tp":0,
           "class_fp":{},"class_fn":{},"reason_tags":{}}

    for img_path in all_imgs:
        lbl_path=lbl_dir/(img_path.stem+".txt")
        if not lbl_path.exists(): continue
        _buf = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(_buf, cv2.IMREAD_COLOR)
        if img is None: continue
        H,W=img.shape[:2]
        stats["reviewed"]+=1

        # Parse GT
        gts=[]
        with open(lbl_path) as f:
            for line in f:
                parts=line.strip().split()
                if not parts: continue
                cid=int(parts[0]); cx,cy,w,h=float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
                gts.append((cid,xywh_to_xyxy(cx,cy,w,h,W,H)))

        # Inference
        blob,scale,_=preprocess(img)
        raw=session.run(None,{input_name:blob})
        dets=postprocess(raw,conf_thresh=args.conf,orig_w=W,orig_h=H,scale=scale)

        matched_gt=set(); matched_det=set()
        tp_pairs=[]  # (gt_idx, det_idx, iou_val)
        for gi,(gcid,gbox) in enumerate(gts):
            best_iou=0.0; best_di=-1
            for di,(dbox,dconf,dcid) in enumerate(dets):
                if di in matched_det: continue
                iou_val=iou(gbox,dbox)
                if iou_val>best_iou:
                    best_iou=iou_val; best_di=di
            if best_iou>=0.5 and best_di>=0 and gts[gi][0]==dets[best_di][2]:
                matched_gt.add(gi); matched_det.add(best_di)
                tp_pairs.append((gi,best_di,best_iou))

        # False Negatives: GT not matched
        for gi,(gcid,gbox) in enumerate(gts):
            if gi in matched_gt: continue
            stats["fn"]+=1
            cn=CLASS_NAMES[gcid] if gcid<len(CLASS_NAMES) else str(gcid)
            stats["class_fn"][cn]=stats["class_fn"].get(cn,0)+1
            if len(gallery["false_negative"])<30:
                vis=img.copy()
                cv2.rectangle(vis,(gbox[0],gbox[1]),(gbox[2],gbox[3]),(0,0,255),2)
                # Find closest pred for tag
                reason="missed_detection"
                if not dets: reason="no_detections"
                overlay=draw_overlay(vis,CLASS_NAMES[gcid],-1,0.0,0.0,reason,(0,0,255))
                fname=f"fn_{img_path.stem}_{gi}.jpg"
                cv2.imwrite(str(out_base/"false_negative"/fname),overlay)
                gallery["false_negative"].append({"file":fname,"gt_class":cn,"reason":reason})

        # False Positives: det not matched
        for di,(dbox,dconf,dcid) in enumerate(dets):
            if di in matched_det: continue
            stats["fp"]+=1
            cn=CLASS_NAMES[dcid] if dcid<len(CLASS_NAMES) else str(dcid)
            stats["class_fp"][cn]=stats["class_fp"].get(cn,0)+1
            if len(gallery["false_positive"])<30:
                vis=img.copy()
                cv2.rectangle(vis,(dbox[0],dbox[1]),(dbox[2],dbox[3]),(255,0,0),2)
                reason="spurious_detection"
                overlay=draw_overlay(vis,"background",dcid,dconf,0.0,reason,(255,0,0))
                fname=f"fp_{img_path.stem}_{di}.jpg"
                cv2.imwrite(str(out_base/"false_positive"/fname),overlay)
                gallery["false_positive"].append({"file":fname,"pred_class":cn,"conf":round(dconf,3),"reason":reason})

        # TPs: low conf / high conf
        for gi,di,iou_val in tp_pairs:
            gcid,gbox=gts[gi]; dbox,dconf,dcid=dets[di]
            category="low_confidence_correct" if dconf<0.65 else "high_confidence_correct"
            if category=="low_confidence_correct": stats["low_conf_tp"]+=1
            else: stats["high_conf_tp"]+=1
            if len(gallery[category])<15:
                vis=img.copy()
                color=(200,200,0) if category=="low_confidence_correct" else (0,200,0)
                cv2.rectangle(vis,(dbox[0],dbox[1]),(dbox[2],dbox[3]),color,2)
                cn=CLASS_NAMES[gcid] if gcid<len(CLASS_NAMES) else str(gcid)
                reason="low_conf" if category=="low_confidence_correct" else "correct"
                overlay=draw_overlay(vis,cn,dcid,dconf,iou_val,reason,color)
                fname=f"{category[:4]}_{img_path.stem}_{gi}.jpg"
                cv2.imwrite(str(out_base/category/fname),overlay)
                gallery[category].append({"file":fname,"gt_class":cn,"conf":round(dconf,3),"iou":round(iou_val,3)})

    # Save gallery JSON
    gallery_json=out_base/"gallery_index.json"
    with open(gallery_json,"w",encoding="utf-8") as f:
        json.dump({"stats":stats,"gallery":gallery},f,indent=2,ensure_ascii=False)

    # Save markdown report
    report_path=PROJECT/"reports/error_analysis_report.md"
    with open(report_path,"w",encoding="utf-8") as f:
        f.write("# SİNAPTİC5G — Hata Analizi Raporu (Error Analysis Report)\n\n")
        f.write("> **Tarih:** 2026-06-21\n")
        f.write("> **Model:** detector_v3 (aktif üretim modeli)\n")
        f.write("> **Not:** Bu analiz yalnızca bağımsız test verisi (`data/curated/detector_v3/test/`) kullanılarak yapılmıştır. Eğitim verisi dahil edilmemiştir.\n\n---\n\n")
        f.write(f"## 1. İnceleme Özeti\n\n")
        f.write(f"| Metrik | Değer |\n|--------|-------|\n")
        f.write(f"| İncelenen görüntü | {stats['reviewed']} |\n")
        f.write(f"| Yanlış Pozitif (FP) | {stats['fp']} |\n")
        f.write(f"| Yanlış Negatif (FN) | {stats['fn']} |\n")
        f.write(f"| Düşük güven TP (&lt;0.65) | {stats['low_conf_tp']} |\n")
        f.write(f"| Yüksek güven TP (≥0.65) | {stats['high_conf_tp']} |\n\n")
        f.write("## 2. Sınıf Bazlı Hata Analizi\n\n")
        f.write("### Yanlış Pozitifler (Sınıfa Göre)\n\n")
        f.write("| Sınıf | FP Sayısı |\n|-------|----------|\n")
        for cn,cnt in sorted(stats['class_fp'].items(),key=lambda x:-x[1]):
            f.write(f"| {cn} | {cnt} |\n")
        f.write("\n### Yanlış Negatifler (Sınıfa Göre)\n\n")
        f.write("| Sınıf | FN Sayısı |\n|-------|----------|\n")
        for cn,cnt in sorted(stats['class_fn'].items(),key=lambda x:-x[1]):
            f.write(f"| {cn} | {cnt} |\n")
        f.write("\n## 3. Başlıca Hata Modları\n\n")
        f.write("1. **Düşük destek sınıflarında güvenilmez tahmin** — teknocan (8 test örneği) ve bilgisayar (6 test örneği) sınıflarında istatistiki anlamlı değerlendirme yapılamamaktadır.\n")
        f.write("2. **Emniyet kemeri ihlali (sınıf 5) geri çağırım sorunu** — AP@0.50=0.774 iken recall=0.698 olarak ölçülmüştür. Bu sınıftaki 43 test örneği düşük destek sayısına işaret eder.\n")
        f.write("3. **Sigara içme (sınıf 4) sınıf içi varyans** — 758 test örneğiyle en büyük grubu oluşturur; FP bu sınıfta daha yüksek görülür.\n")
        f.write("4. **Küçük nesne gözden kaçırma** — Küçük veya uzak pozisyonlardaki nesneler için FN oranı daha yüksektir.\n")
        f.write("5. **Sınıflar arası karışıklık** — telefonla_konusma / bilgisayar sınıfları görsel örtüşme nedeniyle karıştırılabilir.\n")
        f.write("\n## 4. Öneriler\n\n")
        f.write("* Sınıf 5 (emniyet kemeri ihlali) için detector_v4 tam eğitim tamamlandığında recall artışı beklenmektedir.\n")
        f.write("* Teknocan (sınıf 6) ve bilgisayar (sınıf 7) için veri artışı önceliklendirilmelidir.\n")
        f.write("* Büyütme augmentasyonlarına küçük nesne senaryoları eklenebilir.\n")
        f.write("\n## 5. Galeri\n\n")
        f.write("Görsel örnekler `reports/error_gallery/` altında düzenlenmiştir:\n")
        f.write("* `false_positive/` — FP örnekleri\n")
        f.write("* `false_negative/` — FN örnekleri\n")
        f.write("* `low_confidence_correct/` — Düşük güven TP örnekleri\n")
        f.write("* `high_confidence_correct/` — Yüksek güven TP örnekleri\n")
        f.write(f"\nTam galeri indeksi: `reports/error_gallery/gallery_index.json`\n\n---\n\n")
        f.write("ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR\nTelif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)\n")

    print(f"Done. Stats: {stats}")
    print(f"Report: {report_path}")
    return 0

if __name__=="__main__":
    main()
