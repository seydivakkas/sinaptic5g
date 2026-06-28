# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

import sys
from pathlib import Path

# Add ftr_core directory to python path for testing relative imports
ftr_core_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ftr_core_dir))
sys.path.insert(0, str(ftr_core_dir.parent))
print(f"[conftest] Injected path: {ftr_core_dir}")
