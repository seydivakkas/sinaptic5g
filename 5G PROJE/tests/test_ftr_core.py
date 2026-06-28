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

import json
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator

def test_ftr_config_loader():
    """Verify that ftr_config.json contains required class map and thresholds."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "ftr_config.json"
    assert config_path.is_file(), "ftr_config.json is missing!"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    assert "class_map" in config
    assert "thresholds" in config
    assert "default_threshold" in config
    assert "zero_dce_brightness_threshold" in config
    assert "model_registry" in config
    
    # Class map size should match the 9 FTR classes
    assert len(config["class_map"]) == 9
    assert config["class_map"]["6"] == "teknocan"
    assert config["class_map"]["7"] == "bilgisayar"

def test_results_schema_compliance():
    """Validate dummy FTR results.json outputs against the results.schema.json."""
    schema_path = Path(__file__).resolve().parent.parent / "schemas" / "results.schema.json"
    assert schema_path.is_file(), "results.schema.json is missing!"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        
    validator = Draft202012Validator(schema)
    
    # Valid output
    valid_doc = {
        "video_id": "test_video.mp4",
        "arac_bilgisi": {
            "tip": "sedan",
            "plaka": "01A0000",
            "renk": "gri",
            "confidence_score": 0.85
        },
        "tespitler": [
            {
                "zaman_saniye": 1.250,
                "kategori": "sofor_eylemi",
                "etiket": "sigara_icme",
                "confidence_score": 0.92
            },
            {
                "zaman_saniye": 2.500,
                "kategori": "nesneler",
                "etiket": "teknocan",
                "confidence_score": 0.88
            }
        ]
    }
    validator.validate(valid_doc) # Should not raise exception
    
    # Invalid category/label mismatch (sigara_icme cannot be in nesneler category)
    invalid_doc = {
        "video_id": "test_video.mp4",
        "arac_bilgisi": {
            "tip": "sedan",
            "plaka": "01A0000",
            "renk": "gri",
            "confidence_score": 0.85
        },
        "tespitler": [
            {
                "zaman_saniye": 1.250,
                "kategori": "nesneler",
                "etiket": "sigara_icme",
                "confidence_score": 0.92
            }
        ]
    }
    with pytest.raises(Exception):
        validator.validate(invalid_doc)
