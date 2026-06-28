"""
scripts/build_ocr_gt_csv.py
============================
data/ocr_test/labels/*.txt etiket dosyalarından GT CSV olusturur.
Etiket dosyasi formati: <plaka_metni> (tek satir, plaka karakterleri)
CSV formati: image_name,ground_truth_plate
"""
import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OCR_IMG_DIR  = PROJECT / "data/ocr_test/images"
OCR_LBL_DIR  = PROJECT / "data/ocr_test/labels"
OUT_CSV      = PROJECT / "data/raw/ocr_gt_template.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

rows = []
for lbl in sorted(OCR_LBL_DIR.glob("*.txt")):
    txt = lbl.read_text(encoding="utf-8").strip()
    # Etiket dosyasi: ya duz plaka metni ya da YOLO formati
    # YOLO: "class_id cx cy w h" – bu durumda plaka metni dosya adindan alinir
    if txt and not txt.split()[0].isdigit():
        plate = txt.split("\n")[0].strip().upper()
    else:
        # Dosya adindan plaka cikart (orn: "06ABC123.txt" -> "06ABC123")
        plate = lbl.stem.upper().replace("_", "")

    # Eslesen goruntu bul
    for ext in (".jpg", ".png", ".jpeg"):
        img = OCR_IMG_DIR / (lbl.stem + ext)
        if img.is_file():
            rows.append((img.name, plate))
            break

if not rows:
    # Goruntu yoksa dosya adlarini tahmin et
    for lbl in sorted(OCR_LBL_DIR.glob("*.txt")):
        rows.append((lbl.stem + ".jpg", lbl.stem.upper().replace("_", "")))

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_name", "ground_truth_plate"])
    w.writerows(rows)

print(f"GT CSV olusturuldu: {OUT_CSV}")
print(f"Toplam: {len(rows)} kayit")
for r in rows[:8]:
    print(f"  {r[0]:30s} -> {r[1]}")
