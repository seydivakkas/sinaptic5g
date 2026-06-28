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

import sys
import types
from pathlib import Path

# Add ftr_core and final_5g_extension to sys.path so tests can import modules directly
root = Path(__file__).resolve().parent.parent
ftr_core = root / "ftr_core"
final_5g_extension = root / "final_5g_extension"

sys.path.insert(0, str(ftr_core))
sys.path.insert(0, str(final_5g_extension))
sys.path.insert(0, str(final_5g_extension / "mec_backend"))
sys.path.insert(0, str(final_5g_extension / "qod_engine"))
sys.path.insert(0, str(root))

print("[conftest] Injected sys.path:")
print(f"  → {ftr_core}")
print(f"  → {final_5g_extension}")

# Dynamically construct a virtual 'api' package mapping to mec_backend and qod_engine submodules
api_module = types.ModuleType("api")
api_module.__path__ = []
sys.modules["api"] = api_module

for folder in [final_5g_extension / "mec_backend", final_5g_extension / "qod_engine"]:
    if folder.is_dir():
        for py_file in folder.glob("*.py"):
            name = py_file.stem
            if name != "__init__":
                try:
                    import importlib
                    # Import as absolute package to preserve relative imports
                    full_import_name = f"final_5g_extension.{folder.name}.{name}"
                    mod = importlib.import_module(full_import_name)
                    # Register under 'api.<name>'
                    sys.modules[f"api.{name}"] = mod
                    setattr(api_module, name, mod)
                    # Register under '<name>' directly so that 'import auth_utils' also works
                    sys.modules[name] = mod
                except Exception as e:
                    print(f"[conftest] Warning: failed to map api.{name}: {e}")


