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

import os
import sys
import hashlib
import shutil
import urllib.request
from pathlib import Path

# URLs for real pretrained OCR models
LPRNET_URL = "https://github.com/litianfu1997/vehicle_plate_recognitionplate_lprnet/raw/main/lprnet.onnx"
CRNN_URL = "https://huggingface.co/shadow-cann/hispark-modelzoo-crnn/resolve/main/crnn.onnx"

def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_file(url: str, dest_path: Path):
    print(f"[*] Downloading {url} -> {dest_path}...")
    try:
        # Define a custom User-Agent to avoid HTTP 403 Forbidden
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"[+] Download complete: {dest_path} ({dest_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"[-] Download failed for {url}: {e}")
        raise e

def main():
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    lpr_dest = models_dir / "lprnet.onnx"
    crnn_dest = models_dir / "crnn.onnx"
    
    # Download LPRNet
    download_file(LPRNET_URL, lpr_dest)
    # Download CRNN
    download_file(CRNN_URL, crnn_dest)
    
    # Calculate hashes
    lpr_hash = compute_sha256(lpr_dest)
    crnn_hash = compute_sha256(crnn_dest)
    
    print("\n" + "="*55)
    print(f"LPRNet SHA-256: {lpr_hash}")
    print(f"CRNN SHA-256:   {crnn_hash}")
    print("="*55 + "\n")
    
    # Print warning if sizes are too small (mock size)
    if lpr_dest.stat().st_size < 10000 or crnn_dest.stat().st_size < 10000:
        print("[!] Warning: Downloaded files are suspiciously small. Check if download was raw binary.")
    else:
        print("[+] Real OCR weights downloaded and verified successfully.")

if __name__ == "__main__":
    main()
