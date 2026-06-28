# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import logging
from pathlib import Path
from typing import Dict, Any
from src.competition_contract import atomic_write_results

logger = logging.getLogger("sinaptic5g.ftr.result_writer")

class ResultWriter:
    """Safely and atomically writes output json data to target destination."""
    
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

    def write(self, data: Dict[str, Any], schema_path: Path = None) -> Path:
        """Atomically writes FTR results to file path, optionally validating schema first."""
        logger.info("Writing results atomically to: %s", self.output_path)
        return atomic_write_results(data, self.output_path, schema_path)
