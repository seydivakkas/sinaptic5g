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
scripts/rebalance_test_split.py — Val'den Test'e teknocan/bilgisayar tasima
============================================================================
Val split'ten test split'e, teknocan (ID 6) ve bilgisayar (ID 7) sinifi
iceren goruntuler tasinir. Boylece test split'inin istatistiksel gucu artar.

Hedefler:
  - teknocan test: 10 -> ~30 (val'den ~20 ornek tasinir)
  - bilgisayar test:  6 -> ~25 (val'den ~19 ornek tasinir)

Kullanim:
    python scripts/rebalance_test_split.py [--dry-run]
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT / "data" / "curated" / "detector_v5"

TARGET_CLASSES = {6: "teknocan", 7: "bilgisayar"}
# Hedef test sayilari (instance degil, imaj bazli)
TARGET_TEST_IMAGES = {6: 20, 7: 19}  # val'den tasinacak imaj sayisi

SEED = 42


def parse_label_file(label_path: Path) -> list:
    """Label dosyasini okur, satirlari doner."""
    if not label_path.is_file():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return text.split("\n")


def get_class_ids_in_label(label_path: Path) -> set:
    """Bir label dosyasindaki sinif ID'lerini doner."""
    lines = parse_label_file(label_path)
    ids = set()
    for line in lines:
        parts = line.strip().split()
        if parts:
            try:
                ids.add(int(parts[0]))
            except ValueError:
                pass
    return ids


def count_instances(label_dir: Path) -> dict:
    """Her sinif icin toplam instance sayisi."""
    counts = defaultdict(int)
    for f in label_dir.glob("*.txt"):
        for line in parse_label_file(f):
            parts = line.strip().split()
            if parts:
                try:
                    counts[int(parts[0])] += 1
                except ValueError:
                    pass
    return dict(counts)


def find_candidate_images(val_labels_dir: Path, target_class_id: int) -> list:
    """Val'de hedef sinifi iceren imajlari bulur."""
    candidates = []
    for label_file in val_labels_dir.glob("*.txt"):
        class_ids = get_class_ids_in_label(label_file)
        if target_class_id in class_ids:
            candidates.append(label_file.stem)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Val'den Test'e teknocan/bilgisayar tasima")
    parser.add_argument("--dry-run", action="store_true", help="Degisiklik yapmadan kontrol")
    args = parser.parse_args()

    val_images = DATASET_ROOT / "val" / "images"
    val_labels = DATASET_ROOT / "val" / "labels"
    test_images = DATASET_ROOT / "test" / "images"
    test_labels = DATASET_ROOT / "test" / "labels"

    for d in [val_images, val_labels, test_images, test_labels]:
        if not d.is_dir():
            print(f"HATA: Dizin bulunamadi: {d}")
            return 1

    print("=== Val -> Test Rebalance ===")
    print(f"Onceki val instance sayilari: {count_instances(val_labels)}")
    print(f"Onceki test instance sayilari: {count_instances(test_labels)}")
    print()

    random.seed(SEED)
    moved_total = 0
    move_log = []

    for cls_id, cls_name in TARGET_CLASSES.items():
        n_to_move = TARGET_TEST_IMAGES[cls_id]
        candidates = find_candidate_images(val_labels, cls_id)
        random.shuffle(candidates)

        # Sinirla
        to_move = candidates[:n_to_move]
        print(f"Sinif {cls_id} ({cls_name}): {len(candidates)} aday, {len(to_move)} tasinacak")

        for stem in to_move:
            # Olasi uzantilar
            img_moved = False
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                src_img = val_images / f"{stem}{ext}"
                if src_img.is_file():
                    dst_img = test_images / f"{stem}{ext}"
                    if not args.dry_run:
                        shutil.move(str(src_img), str(dst_img))
                    img_moved = True
                    break

            src_lbl = val_labels / f"{stem}.txt"
            dst_lbl = test_labels / f"{stem}.txt"

            if src_lbl.is_file() and img_moved:
                if not args.dry_run:
                    shutil.move(str(src_lbl), str(dst_lbl))
                moved_total += 1
                move_log.append({
                    "stem": stem,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "classes_in_image": list(get_class_ids_in_label(
                        dst_lbl if not args.dry_run else src_lbl
                    ))
                })

    print(f"\nToplam tasinan imaj: {moved_total}")

    if args.dry_run:
        print("[DRY-RUN] Hicbir dosya tasinmadi.")
    else:
        # labels.cache dosyalarini sil (yeniden olusturulacak)
        for cache in [val_labels.parent / "labels.cache", test_labels.parent / "labels.cache"]:
            if cache.is_file():
                cache.unlink()
                print(f"Silindi: {cache}")

    # Sonrasi sayim
    if not args.dry_run:
        new_val_counts = count_instances(val_labels)
        new_test_counts = count_instances(test_labels)
        print(f"\nYeni val instance sayilari: {new_val_counts}")
        print(f"Yeni test instance sayilari: {new_test_counts}")

        # Val ve test imaj sayilari
        val_img_count = len(list(val_images.iterdir()))
        test_img_count = len(list(test_images.iterdir()))

        # Split ozeti kaydet
        summary = {
            "train": {
                "images": len(list((DATASET_ROOT / "train" / "images").iterdir())),
                "labels": len(list((DATASET_ROOT / "train" / "labels").glob("*.txt"))),
                "instances": count_instances(DATASET_ROOT / "train" / "labels")
            },
            "val": {
                "images": val_img_count,
                "labels": len(list(val_labels.glob("*.txt"))),
                "instances": new_val_counts
            },
            "test": {
                "images": test_img_count,
                "labels": len(list(test_labels.glob("*.txt"))),
                "instances": new_test_counts
            },
            "moved_images": moved_total,
            "move_log": move_log
        }

        report_path = PROJECT / "reports" / "detector_v5_split_summary_v2.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSplit ozeti kaydedildi: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
