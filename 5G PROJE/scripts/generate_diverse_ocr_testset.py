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
scripts/generate_diverse_ocr_testset.py — 100+ Çeşitli Sentetik Plaka Üretici
=============================================================================
Faz 3: OCR Test Seti Çeşitlendirme

Gerçek dünya doğrulaması için 100+ çeşitli Türk plakası formatı üretir.
- 01-81 tüm il kodları
- Farklı harf/rakam kombinasyonları (e.g. 34ABC123, 06A1234, 35XY789)
- Görsel bozulmalar (Bulanıklık, Gürültü, Perspektif Kayması, Parlaklık Değişimi)
- data/ocr_test_v2/images/ ve data/ocr_test_v2/labels/ altına kaydeder.
"""

import cv2
import numpy as np
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_IMAGES_DIR = PROJECT_ROOT / "data" / "ocr_test_v2" / "images"
OUT_LABELS_DIR = PROJECT_ROOT / "data" / "ocr_test_v2" / "labels"

# Dizinleri oluştur
OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

TURKISH_PLATE_PATTERN = re.compile(r'^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}[0-9]{2,4}$')
VOCABULARY = "-" + "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Türk plaka format şablonları
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBERS = "0123456789"

def generate_random_plate_text() -> str:
    """Rastgele ve geçerli bir Türk plaka yazısı üretir."""
    # İl kodu (01-81)
    city_code = f"{random.randint(1, 81):02d}"
    
    # Format seçimi
    # 1: 06 A 1234
    # 2: 06 AB 123
    # 3: 06 ABC 12
    # 4: 06 AB 1234
    # 5: 06 ABC 123
    fmt = random.choice([1, 2, 3, 4, 5])
    
    if fmt == 1:
        letters = "".join(random.choice(LETTERS) for _ in range(1))
        digits = "".join(random.choice(NUMBERS) for _ in range(4))
    elif fmt == 2:
        letters = "".join(random.choice(LETTERS) for _ in range(2))
        digits = "".join(random.choice(NUMBERS) for _ in range(3))
    elif fmt == 3:
        letters = "".join(random.choice(LETTERS) for _ in range(3))
        digits = "".join(random.choice(NUMBERS) for _ in range(2))
    elif fmt == 4:
        letters = "".join(random.choice(LETTERS) for _ in range(2))
        digits = "".join(random.choice(NUMBERS) for _ in range(4))
    else:  # fmt == 5
        letters = "".join(random.choice(LETTERS) for _ in range(3))
        digits = "".join(random.choice(NUMBERS) for _ in range(3))
        
    plate = f"{city_code}{letters}{digits}"
    if TURKISH_PLATE_PATTERN.match(plate):
        return plate
    return "34TR1000"  # Fallback

def draw_plate_image(text: str) -> np.ndarray:
    """Temiz bir plaka resmi çizer (160x32)."""
    img = np.ones((32, 160, 3), dtype=np.uint8) * 255
    
    # 1. Sol tarafa mavi şerit çiz (TR logosu için)
    cv2.rectangle(img, (0, 0), (14, 32), (255, 0, 0), -1)
    
    # Mavi şerit üzerine ince beyaz TR yazısı çizilebilir (opsiyonel)
    cv2.putText(img, "TR", (1, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 2. Plaka çerçevesi çiz (siyah ince kenarlık)
    cv2.rectangle(img, (0, 0), (159, 31), (0, 0, 0), 1)
    
    # 3. Metni yerleştir
    # Yazıyı ortalamak için boyuta göre konum ayarla
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    
    # Plaka formatına göre karakter boşluğunu simüle etmek için boşluklu çizelim
    # Örn: "34 TR 123" gibi görünecek ama etiket "34TR123" olacak
    # İl kodundan sonra ve harflerden sonra boşluk koyarak çizelim
    match = re.match(r'^([0-9]{2})([A-Z]+)([0-9]+)$', text)
    if match:
        drawn_text = f"{match.group(1)} {match.group(2)} {match.group(3)}"
    else:
        drawn_text = text
        
    text_size = cv2.getTextSize(drawn_text, font, font_scale, thickness)[0]
    # Kalan boşluğu ortala
    x = 16 + (144 - text_size[0]) // 2
    y = 23
    
    cv2.putText(img, drawn_text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img

def apply_augmentations(img: np.ndarray) -> np.ndarray:
    """Gerçek dünya koşullarını simüle etmek için rastgele bozulmalar ekler."""
    out = img.copy()
    h, w = out.shape[:2]
    
    # A. Perspektif kayması / Döndürme (%50 ihtimalle)
    if random.random() < 0.5:
        # Rastgele küçük kayma değerleri
        dx = random.randint(-4, 4)
        dy = random.randint(-2, 2)
        
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [0 + dx, 0 + dy],
            [w - dx, 0 - dy],
            [0 - dx, h - dy],
            [w + dx, h + dy]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        out = cv2.warpPerspective(out, matrix, (w, h), borderValue=(255, 255, 255))
        
    # B. Parlaklık / Kontrast Değişimi (%70 ihtimalle)
    if random.random() < 0.7:
        alpha = random.uniform(0.7, 1.3)  # Kontrast
        beta = random.randint(-30, 30)    # Parlaklık
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
        
    # C. Bulanıklık (Blur) (%50 ihtimalle)
    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)
        
    # D. Gaussian Gürültü (%40 ihtimalle)
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(5, 15), out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    return out

def main():
    print("=== OCR Çeşitlendirilmiş Test Seti Üretici ===")
    
    count = 110  # 100+ plaka
    generated_plates = set()
    
    # 1. 01-81 arası her ilden en az bir plaka üretelim ki tüm Türkiye'yi kapsasın
    for city_idx in range(1, 82):
        city_code = f"{city_idx:02d}"
        letters = "".join(random.choice(LETTERS) for _ in range(random.choice([1, 2, 3])))
        digits = "".join(random.choice(NUMBERS) for _ in range(random.choice([2, 3, 4])))
        plate = f"{city_code}{letters}{digits}"
        if TURKISH_PLATE_PATTERN.match(plate):
            generated_plates.add(plate)
            
    # 2. Kalanları 110'a tamamlayana kadar rastgele üret
    while len(generated_plates) < count:
        generated_plates.add(generate_random_plate_text())
        
    plates_list = list(generated_plates)
    random.shuffle(plates_list)
    
    print(f"Toplam üretilecek benzersiz plaka sayısı: {len(plates_list)}")
    
    for idx, plate_text in enumerate(plates_list):
        img_clean = draw_plate_image(plate_text)
        img_aug = apply_augmentations(img_clean)
        
        img_path = OUT_IMAGES_DIR / f"plate_{idx+1:03d}.jpg"
        lbl_path = OUT_LABELS_DIR / f"plate_{idx+1:03d}.txt"
        
        # Dosyaya kaydet (Unicode uyumlu)
        _, buf = cv2.imencode(".jpg", img_aug)
        buf.tofile(str(img_path))
        
        lbl_path.write_text(plate_text, encoding="utf-8")
        
        if (idx + 1) % 20 == 0 or (idx + 1) == len(plates_list):
            print(f"  [{idx+1}/{len(plates_list)}] Kaydedildi: {plate_text}")
            
    print(f"Başarıyla {len(plates_list)} adet çeşitlendirilmiş plaka görseli ve etiketi oluşturuldu.")
    print(f"Dizin: {OUT_IMAGES_DIR}")
    return 0

if __name__ == "__main__":
    main()
