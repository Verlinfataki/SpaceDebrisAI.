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

# ==========================================================
# Débris spatiaux
# ==========================================================


CELESTRAK_FENGYUN_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=json"
)

CELESTRAK_IRIDIUM33_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=IRIDIUM-33-DEBRIS&FORMAT=json"
)

CELESTRAK_COSMOS2251_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=COSMOS-2251-DEBRIS&FORMAT=json"
)

# En-têtes HTTP
HEADERS = {
    "User-Agent": "SpaceDebrisAI/1.0"
}

# ==========================================================
# Rapports
# ==========================================================

REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"

TABLES_DIR = REPORTS_DIR / "tables"

LOGS_DIR = REPORTS_DIR / "logs"

FEATURE_IMPORTANCE_FILE = (
    TABLES_DIR /
    "feature_importance.csv"
)

FEATURE_IMPORTANCE_FIGURE = (
    FIGURES_DIR /
    "feature_importance.png"
)

# ==========================================================
# Jeux de données d'entraînement
# ==========================================================

DATASET_SOURCES = [
    {
        "filename": "active_satellites.json",
        "url": CELESTRAK_ACTIVE_URL,
        "label": "Satellite",
    },
    {
        "filename": "fengyun1c_debris.json",
        "url": CELESTRAK_FENGYUN_URL,
        "label": "Debris",
    },
    {
        "filename": "iridium33_debris.json",
        "url": CELESTRAK_IRIDIUM33_URL,
        "label": "Debris",
    },
    {
        "filename": "cosmos2251_debris.json",
        "url": CELESTRAK_COSMOS2251_URL,
        "label": "Debris",
    },
]

# ==========================================================
# Fichiers de données
# ==========================================================

TRAINING_DATASET_FILE = (
    PROCESSED_DATA_DIR / "training_dataset.csv"
)

CLEAN_DATASET_FILE = (
    PROCESSED_DATA_DIR / "clean_training_dataset.csv"
)

# ==========================================================
# Variables pour l'analyse et le Machine Learning
# ==========================================================

# Variables numériques
NUMERICAL_FEATURES = [
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
]

TOP_5_FEATURES = [
    "ECCENTRICITY",
    "INCLINATION",
    "MEAN_MOTION",
    "BSTAR",
    "MEAN_MOTION_DOT",
]

TOP_3_FEATURES = [
    "ECCENTRICITY",
    "INCLINATION",
    "MEAN_MOTION",
]

# Variables supprimées avant l'entraînement
DROPPED_FEATURES = [
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
]

# Variable cible
TARGET_COLUMN = "LABEL"

# ==========================================================
# Paramètres statistiques
# ==========================================================

SIGNIFICANCE_LEVEL = 0.05