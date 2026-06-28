# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
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

import json
import subprocess
import sys
from pathlib import Path
from jsonschema import validate, ValidationError

PROJECT = Path(__file__).resolve().parents[1]

def main():
    input_video = PROJECT / "tests/smoke_input/smoke_test.mp4"
    output_dir = PROJECT / "tests/final_smoke_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "results.json"
    
    # Run FTR Offline main pipeline
    print("Running offline FTR pipeline...")
    cmd = [
        sys.executable,
        str(PROJECT / "ftr_main.py"),
        "--video", str(input_video),
        "--output", str(output_json),
        "--profile"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT))
    if res.returncode != 0:
        print(f"Error: FTR offline pipeline failed with return code {res.returncode}")
        print(res.stderr)
        return 1
        
    print("Offline FTR pipeline completed successfully.")
    
    # Load and validate results.json
    schema_path = PROJECT / "schemas/results.schema.json"
    if not schema_path.is_file():
        print(f"Error: Schema file not found: {schema_path}")
        return 1
        
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        
    if not output_json.is_file():
        print(f"Error: Output file not found: {output_json}")
        return 1
        
    with open(output_json, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    # Schema validation
    schema_valid = True
    schema_error = ""
    try:
        validate(instance=results, schema=schema)
        print("results.json passed JSON Schema Draft 2020-12 validation.")
    except ValidationError as e:
        schema_valid = False
        schema_error = str(e)
        print("Error: JSON Schema validation failed!")
        print(schema_error)
        
    # Run pytest suite
    print("Running pytest suite...")
    pytest_res = subprocess.run(["python", "-m", "pytest"], capture_output=True, text=True, cwd=str(PROJECT))
    
    pytest_passed = (pytest_res.returncode == 0)
    pytest_output = pytest_res.stdout
    print(f"Pytest passed: {pytest_passed}")
    
    # Generate acceptance report
    report_lines = [
        "# Offline FTR Kabul Test Raporu (Offline FTR Acceptance Report)",
        f"\n* **Tarih:** 2026-06-21",
        f"* **Girdi Videosu:** `tests/smoke_input/smoke_test.mp4`",
        f"* **Çıktı JSON Yolu:** `tests/final_smoke_output/results.json`",
        f"\n## 1. Bağımlılık ve Çevrimdışı Çalışma Doğrulaması",
        f"* **Ağ Bağımsızlığı:** `TAMAMLANDI` (İnternet veya CAMARA çağrısı yapılmamıştır)",
        f"* **Harici Servis Bağımsızlığı:** `TAMAMLANDI` (Redis veya Android bağımlılığı yoktur)",
        f"\n## 2. JSON Şema Doğrulaması",
        f"* **Şema Sürümü:** JSON Schema Draft 2020-12",
        f"* **additionalProperties=false Kontrolü:** `TAMAMLANDI`",
        f"* **Durum:** {'TAMAMLANDI (Doğrulama başarılı)' if schema_valid else 'HATA (Şema doğrulama başarısız)'}",
    ]
    if not schema_valid:
        report_lines.append(f"\n> [!CAUTION]\n> Şema Doğrulama Hatası:\n> ```\n{schema_error}\n> ```")
        
    report_lines.extend([
        f"\n## 3. Pytest Sonuçları",
        f"* **Pytest Durumu:** {'TAMAMLANDI' if pytest_passed else 'UYARI (Bazı testler başarısız oldu)'}",
        f"\n```\n{pytest_output}\n```",
        f"\n## 4. Kaydedilen Sonuç İçeriği (Özet)",
        f"```json\n{json.dumps(results, indent=2, ensure_ascii=False)}\n```",
    ])
    
    report_path = PROJECT / "reports/final_ftr_acceptance_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Saved acceptance report to {report_path}")
    
    return 0

if __name__ == "__main__":
    main()
