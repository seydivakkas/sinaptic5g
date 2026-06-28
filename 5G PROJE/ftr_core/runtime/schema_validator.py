# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import json
import logging
from pathlib import Path
from typing import Dict, Any
from jsonschema import Draft202012Validator

logger = logging.getLogger("sinaptic5g.ftr.schema_validator")

class SchemaValidator:
    """Validates output FTR JSON data against schemas/results.schema.json."""
    
    def __init__(self, schema_path: Path):
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        if not self.schema_path.is_file():
            # Try fallback to parent directory structure
            alt_path = Path(__file__).resolve().parent.parent.parent / "schemas" / "results.schema.json"
            if alt_path.is_file():
                self.schema_path = alt_path
            else:
                raise FileNotFoundError(f"JSON schema file is missing: {self.schema_path}")
                
        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        Draft202012Validator.check_schema(schema)
        return schema

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validates result dict against results.schema.json. Raises jsonschema.ValidationError if invalid."""
        Draft202012Validator(self.schema).validate(data)
        logger.info("Output results.json validated successfully against schema.")
        return True
