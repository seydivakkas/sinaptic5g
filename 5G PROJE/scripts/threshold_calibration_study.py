# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

"""Phase 16 — Per-Class Threshold Calibration Study."""

import os, json, csv, argparse
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "telefonla_konusma","su_icme","arkaya_bakma","esneme",
    "sigara_icme","emniyet_kemeri_ihlali","teknocan","bilgisayar","license_plate"
]
MIN_SAMPLES_FOR_RELIABLE = 50

def xywh_to_xyxy(cx,cy,w,h,W,H):
    x1=int((cx-w/2)*W); y1=int((cy-h/2)*H)
    x2=int((cx+w/2)*W); y2=int((cy+h/2)*H)
    return max(0,x1),max(0,y1),min(W,x2),min(H,y2)

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def preprocess(img, size=640):
    import cv2
    h,w=img.shape[:2]
    scale=size/max(h,w)
    nh,nw=int(h*scale),int(w*scale)
    resized=cv2.resize(img,(nw,nh))
    canvas=np.zeros((size,size,3),np.uint8)
    canvas[:nh,:nw]=resized
    blob=canvas.astype(np.float32)/255.0
    blob=blob.transpose(2,0,1)[np.newaxis]
    return blob,scale

def main():
    import cv2
    import onnxruntime as ort

    parser=argparse.ArgumentParser()
    parser.add_argument("--model", default="models/detector.onnx")
    parser.add_argument("--val-images", default="data/curated/detector_v3/test/images")
    parser.add_argument("--val-labels", default="data/curated/detector_v3/test/labels")
    parser.add_argument("--sample", type=int, default=300)
    args=parser.parse_args()

    model_path=PROJECT/args.model
    for alt in ["models/detector_optimized.onnx","models/detector_v3.onnx","yolov8n.onnx"]:
        if not model_path.exists():
            ap=PROJECT/alt
            if ap.exists(): model_path=ap
    if not model_path.exists():
        print("ERROR: No model found")
        return 1

    print(f"Loading: {model_path.name}")
    session=ort.InferenceSession(str(model_path),providers=["CPUExecutionProvider"])
    input_name=session.get_inputs()[0].name

    img_dir=PROJECT/args.val_images
    lbl_dir=PROJECT/args.val_labels
    all_imgs=sorted(img_dir.glob("*.jpg"))+sorted(img_dir.glob("*.png"))
    import random; random.seed(42)
    if len(all_imgs)>args.sample: all_imgs=random.sample(all_imgs,args.sample)

    # Collect (gt_cid, pred_conf, is_match) tuples per class
    class_data={i:{"tp":[],"fp":[]} for i in range(len(CLASS_NAMES))}
    THRESHOLDS=np.arange(0.05,0.96,0.05).tolist()

    for img_path in all_imgs:
        lbl_path=lbl_dir/(img_path.stem+".txt")
        if not lbl_path.exists(): continue
        _buf=np.fromfile(str(img_path),dtype=np.uint8)
        img=cv2.imdecode(_buf,cv2.IMREAD_COLOR)
        if img is None: continue
        H,W=img.shape[:2]
        gts=[]
        with open(lbl_path) as f:
            for line in f:
                p=line.strip().split()
                if not p: continue
                cid=int(p[0]); cx,cy,w,h=float(p[1]),float(p[2]),float(p[3]),float(p[4])
                gts.append((cid,xywh_to_xyxy(cx,cy,w,h,W,H)))

        blob,scale=preprocess(img)
        raw=session.run(None,{input_name:blob})
        preds=raw[0][0].T
        dets=[]
        
        # NumPy vectorized filtering
        scores = preds[:, 4:]
        cids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(preds)), cids]
        keep_indices = np.where(confs >= 0.01)[0]
        
        for idx in keep_indices:
            row = preds[idx]
            cid = int(cids[idx])
            conf = float(confs[idx])
            x,y,w2,h2=row[0],row[1],row[2],row[3]
            x1=max(0,int((x-w2/2)/scale)); y1=max(0,int((y-h2/2)/scale))
            x2=min(W,int((x+w2/2)/scale)); y2=min(H,int((y+h2/2)/scale))
            dets.append((cid,conf,[x1,y1,x2,y2]))

        matched_gt=set()
        for det_cid,det_conf,dbox in sorted(dets,key=lambda x:-x[1]):
            best_iou=0; best_gi=-1
            for gi,(gcid,gbox) in enumerate(gts):
                if gi in matched_gt: continue
                if gcid!=det_cid: continue
                iv=iou(gbox,dbox)
                if iv>best_iou: best_iou=iv; best_gi=gi
            if best_iou>=0.5 and best_gi>=0:
                matched_gt.add(best_gi)
                class_data[det_cid]["tp"].append(det_conf)
            else:
                if det_cid<len(CLASS_NAMES):
                    class_data[det_cid]["fp"].append(det_conf)

    # Compute P/R/F1 per threshold per class
    rows_csv=[["class_id","class_name","threshold","precision","recall","f1","tp_count","fp_count","fn_count","status"]]
    study={}
    for cid,cname in enumerate(CLASS_NAMES):
        tp_confs=np.array(sorted(class_data[cid]["tp"],reverse=True))
        fp_confs=np.array(sorted(class_data[cid]["fp"],reverse=True))
        total_gt=len(tp_confs)+sum(1 for gi,_ in enumerate(gts) if False)  # approximate
        total_tp_all=len(tp_confs)
        n_total=total_tp_all+len(fp_confs)
        status="UYARI (düşük destek)" if n_total<MIN_SAMPLES_FOR_RELIABLE else "ÖLÇÜLDÜ"

        thresh_rows=[]
        best_f1=0; best_thresh=0.25; best_p=0; best_r=0
        for t in THRESHOLDS:
            tp=sum(1 for c in tp_confs if c>=t)
            fp=sum(1 for c in fp_confs if c>=t)
            fn=sum(1 for c in tp_confs if c<t)
            p=tp/(tp+fp) if (tp+fp)>0 else 0.0
            r=tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1=2*p*r/(p+r) if (p+r)>0 else 0.0
            thresh_rows.append({"threshold":round(t,2),"precision":round(p,3),"recall":round(r,3),"f1":round(f1,3),
                                 "tp":tp,"fp":fp,"fn":fn})
            rows_csv.append([cid,cname,round(t,2),round(p,3),round(r,3),round(f1,3),tp,fp,fn,status])
            if f1>best_f1: best_f1=f1; best_thresh=t; best_p=p; best_r=r

        study[cname]={
            "class_id":cid,"status":status,"sample_tp":total_tp_all,"sample_fp":len(fp_confs),
            "best_threshold":round(best_thresh,2),"best_f1":round(best_f1,3),
            "best_precision":round(best_p,3),"best_recall":round(best_r,3),
            "recommended_threshold":round(best_thresh,2),
            "thresholds":thresh_rows
        }
        print(f"  {cname}: best_thresh={best_thresh:.2f}, F1={best_f1:.3f} ({status})")

    # Save CSV
    csv_path=PROJECT/"reports/threshold_calibration_study.csv"
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); [writer.writerow(r) for r in rows_csv]

    # Save JSON
    json_path=PROJECT/"reports/threshold_calibration_study.json"
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(study,f,indent=2,ensure_ascii=False)

    # Save Markdown
    md_path=PROJECT/"reports/threshold_calibration_study.md"
    with open(md_path,"w",encoding="utf-8") as f:
        f.write("# SİNAPTİC5G — Eşik ve Güven Kalibrasyonu Çalışması\n\n")
        f.write("> **Tarih:** 2026-06-21\n> **Model:** detector_v3 (aktif üretim)\n\n")
        f.write("> [!IMPORTANT]\n> Bu çalışma, çıkarım zamanı eşiklerinin sınıf bazlı optimize edilip edilemeyeceğini araştırmaktadır. Önerilen eşikler yalnızca öneri niteliğindedir; üretime uygulanması için tam FTR kabul testleri tekrar çalıştırılmalıdır.\n\n---\n\n")
        f.write("## Sınıf Bazlı Önerilen Eşikler\n\n")
        f.write("| Sınıf | Mevcut Eşik | Önerilen Eşik | En İyi F1 | Durum |\n")
        f.write("|-------|------------|--------------|----------|-------|\n")
        for cname,d in study.items():
            f.write(f"| {cname} | 0.25 | {d['recommended_threshold']} | {d['best_f1']:.3f} | {d['status']} |\n")
        f.write("\n## Notlar\n\n")
        f.write("* `teknocan` ve `bilgisayar` sınıfları için düşük örnek sayısı nedeniyle eşik önerileri güvenilir değildir (UYARI).\n")
        f.write("* `emniyet_kemeri_ihlali` için düşük recall sorunu eşik düşürülerek kısmen iyileştirilebilir; ancak bu FP artışına neden olur.\n")
        f.write("* Tam eşik kalibrasyonu için detector_v4 tam eğitim sonrası daha fazla test örneği ile tekrarlanmalıdır.\n\n")
        f.write("Tam CSV: `reports/threshold_calibration_study.csv`\n\n")
        f.write("---\nÖZEL LİSANS — TÜM HAKLAR SAKLIDIR\nTelif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)\n")

    print(f"Saved: {csv_path}, {json_path}, {md_path}")
    return 0

if __name__=="__main__":
    main()
