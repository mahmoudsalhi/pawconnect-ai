"""Re-export of backend/app/ai/scam_text.py, the single source of the text preparation.

Training and the app must prepare text identically, so the code lives in the backend (where the model
runs) and is loaded here by file path (the backend package itself is not imported).
"""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "backend" / "app" / "ai" / "scam_text.py"
_spec = importlib.util.spec_from_file_location("scam_text", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

mask_contacts = _mod.mask_contacts
normalize = _mod.normalize
prepare = _mod.prepare
