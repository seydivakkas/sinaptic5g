# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
#
# Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
# amaçlı olarak paylaşılmıştır.

"""qod_atexit_patch - Stands alone script to validate results.json against results.schema.json."""

import os
import sys
import json
from pathlib import Path

def validate_results(results_path: Path, schema_path: Path) -> bool:
    if not results_path.is_file():
        print(f"[-] Error: results.json not found at {results_path}", file=sys.stderr)
        return False
    if not schema_path.is_file():
        print(f"[-] Error: schema not found at {schema_path}", file=sys.stderr)
        return False

    try:
        from jsonschema import validate
        from jsonschema.exceptions import ValidationError
    except ImportError:
        print("[-] Error: jsonschema python package is not installed.", file=sys.stderr)
        return False

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
            
        validate(instance=data, schema=schema)
        print(f"[+] Success: {results_path.name} is valid against {schema_path.name}")
        return True
    except ValidationError as ve:
        print(f"[-] Schema Validation Error: {ve.message}", file=sys.stderr)
        print(f"[-] Error path: {list(ve.absolute_path)}", file=sys.stderr)
        return False
    except json.JSONDecodeError as jde:
        print(f"[-] JSON Decode Error: {jde}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[-] Error during validation: {e}", file=sys.stderr)
        return False

def main():
    default_results = Path("data/output/results.json")
    if "FTR_OUTPUT_PATH" in os.environ:
        default_results = Path(os.environ["FTR_OUTPUT_PATH"])
        
    schema_path = Path("schemas/results.schema.json")
    if not schema_path.is_file():
        # fallback path search
        schema_path = Path(__file__).parent.parent / "schemas" / "results.schema.json"

    # Allow custom path via args
    results_path = default_results
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])

    success = validate_results(results_path, schema_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
