# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

"""
Phase 18 — OCR Accuracy Evaluation Kit.
Provides an evaluation pipeline for license plate OCR character accuracy (CER / EM).
Reads a GT CSV and runs LPRNet/CRNN to compute metrics.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT))

from plate_ocr import PlateRecognizer

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_metrics(predictions, ground_truths):
    total_dist = 0
    total_chars = 0
    exact_matches = 0
    total_samples = len(predictions)
    
    for pred, gt in zip(predictions, ground_truths):
        pred_clean = pred.strip().upper()
        gt_clean = gt.strip().upper()
        
        dist = levenshtein_distance(pred_clean, gt_clean)
        total_dist += dist
        total_chars += len(gt_clean)
        if pred_clean == gt_clean:
            exact_matches += 1
            
    cer = total_dist / max(1, total_chars)
    em = exact_matches / max(1, total_samples)
    return cer, em

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-csv", default="data/raw/ocr_gt_template.csv")
    parser.add_argument("--image-dir", default="data/raw/ocr_test_images")
    parser.add_argument("--report-md", default="reports/ocr_accuracy_evaluation.md")
    args = parser.parse_args()
    
    gt_csv_path = PROJECT / args.gt_csv
    image_dir_path = PROJECT / args.image_dir
    report_path = PROJECT / args.report_md
    
    # 1. Create directory structure and template if missing
    gt_csv_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir_path.mkdir(parents=True, exist_ok=True)
    
    if not gt_csv_path.exists():
        with open(gt_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "ground_truth_plate"])
            writer.writerow(["sample_plate_01.jpg", "06AAA001"])
            writer.writerow(["sample_plate_02.jpg", "34BBB999"])
        print(f"Created template GT CSV: {gt_csv_path}")
        
    print("Initializing PlateRecognizer...")
    recognizer = PlateRecognizer()
    
    # Check if there are real test images in the directory
    test_files = list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.png"))
    
    if not test_files:
        print(f"No test images found in {image_dir_path}. Running dry-run validation with dummy data.")
        # Dry-run validation
        predictions = ["06AAA001", "34BBB998", "35CCC123"]
        ground_truths = ["06AAA001", "34BBB999", "35CCC123"]
        
        cer, em = calculate_metrics(predictions, ground_truths)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# SİNAPTİC5G — OCR Karakter Doğruluğu Değerlendirme Raporu\n\n")
            f.write("> **Tarih:** 2026-06-21\n")
            f.write("> **Tür:** Kuru Çalıştırma (Dry-run / Hazırlık Kiti)\n\n")
            f.write("> [!WARNING]\n")
            f.write(f"> Gerçek plaka test görüntüleri bulunamadığından, test simüle edilmiş veri ile gerçekleştirilmiştir. Gerçek ölçüm için `{args.image_dir}` dizinine görüntüleri yükleyip `{args.gt_csv}` şablon dosyasını doldurunuz.\n\n")
            f.write("---\n\n## Simüle Edilen Performans Metrikleri\n\n")
            f.write(f"* **Toplam Örnek Sayısı:** {len(predictions)}\n")
            f.write(f"* **Karakter Hata Oranı (CER):** {cer:.2%} (Hedef: < 5%)\n")
            f.write(f"* **Tam Eşleşme Doğruluğu (Exact Match EM):** {em:.2%} (Hedef: > 90%)\n\n")
            f.write("## Değerlendirme ve Notlar\n")
            f.write("1. Plaka okuyucu (PlateRecognizer) CTC greedy decoding tabanlı CRNN mimarisi kullanmaktadır.\n")
            f.write("2. Türkçe plaka format kuralları (`TURKISH_PLATE_PATTERN`) sayesinde geçersiz okumalar elenmekte veya düzeltilmektedir.\n\n")
            f.write("---\n\nÖZEL LİSANS — TÜM HAKLAR SAKLIDIR\nTelif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)\n")
        print(f"Saved dry-run report to {report_path}")
        return 0
        
    # Real validation if files exist
    gt_map = {}
    with open(gt_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                gt_map[row[0].strip()] = row[1].strip()
                
    predictions = []
    ground_truths = []
    
    for img_path in test_files:
        name = img_path.name
        if name not in gt_map:
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        res = recognizer.recognize(img)
        pred_text = res[0] if res else ""
        
        predictions.append(pred_text)
        ground_truths.append(gt_map[name])
        print(f"Processed: {name} | GT: {gt_map[name]} | Pred: {pred_text}")
        
    if not predictions:
        print("ERROR: No matching files found between CSV manifest and test directory.")
        return 1
        
    cer, em = calculate_metrics(predictions, ground_truths)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SİNAPTİC5G — OCR Karakter Doğruluğu Değerlendirme Raporu\n\n")
        f.write("> **Tarih:** 2026-06-21\n")
        f.write("> **Tür:** Gerçek Veri Ölçümü\n\n")
        f.write("---\n\n## Performans Metrikleri\n\n")
        f.write(f"* **Toplam Örnek Sayısı:** {len(predictions)}\n")
        f.write(f"* **Karakter Hata Oranı (CER):** {cer:.4%}\n")
        f.write(f"* **Tam Eşleşme Doğruluğu (Exact Match EM):** {em:.4%}\n\n")
        f.write("## Detaylı Sonuçlar\n\n")
        f.write("| Görüntü Adı | Ground Truth | Tahmin | Levenshtein Mesafesi |\n")
        f.write("|---|---|---|---|\n")
        for idx, img_path in enumerate(test_files):
            name = img_path.name
            if name in gt_map:
                dist = levenshtein_distance(predictions[idx], ground_truths[idx])
                f.write(f"| {name} | {ground_truths[idx]} | {predictions[idx]} | {dist} |\n")
        f.write("\n---\n\nÖZEL LİSANS — TÜM HAKLAR SAKLIDIR\nTelif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)\n")
        
    print(f"Saved real validation report to {report_path}")
    return 0

if __name__ == "__main__":
    main()
