"""
Configuration globale du projet SpaceDebrisAI
"""

from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dossiers
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# URL CelesTrak
CELESTRAK_ACTIVE_URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?GROUP=active&FORMAT=json"
)

# En-têtes HTTP
HEADERS = {
    "User-Agent": "SpaceDebrisAI/1.0"
}