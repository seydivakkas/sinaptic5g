# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import json
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator

def test_results_schema_loading_and_validation():
    schema_path = Path(__file__).resolve().parent.parent.parent / "schemas" / "results.schema.json"
    assert schema_path.is_file(), "Schema results.schema.json not found"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        
    validator = Draft202012Validator(schema)
    
    valid_results = {
        "video_id": "eval.mp4",
        "arac_bilgisi": {
            "tip": "sedan",
            "plaka": "34ABC123",
            "renk": "mavi",
            "confidence_score": 0.82
        },
        "tespitler": [
            {
                "zaman_saniye": 0.50,
                "kategori": "sofor_eylemi",
                "etiket": "esneme",
                "confidence_score": 0.90
            }
        ]
    }
    
    validator.validate(valid_results)
    
    # Invalid plate pattern check
    invalid_results = valid_results.copy()
    invalid_results["arac_bilgisi"] = {
        "tip": "sedan",
        "plaka": "99XX9999", # Invalid Turkish plate code 99
        "renk": "mavi",
        "confidence_score": 0.82
    }
    
    with pytest.raises(Exception):
        validator.validate(invalid_results)
