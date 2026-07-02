"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : persistence.py

Description :
Sauvegarde et charge les modèles
de Machine Learning.

Responsabilités :
    - Sauvegarder un modèle
    - Charger un modèle

============================================================
"""
# ==========================================================
# Bibliothèques tierces
# ==========================================================
import joblib
from pathlib import Path

# ==========================================================
# Modules du projet
# ==========================================================
from src.config import (
    MODELS_DIR,
    RANDOM_FOREST_MODEL,
    CLEAN_DATASET_FILE,
    TARGET_COLUMN,
    TOP_3_FEATURES,
)
from src.data.io import load_csv_dataset
from src.features.pipeline import prepare_dataset
from src.models.training import train_random_forest


def save_model(
    model,
    filename: str,
) -> Path:
    """
    Sauvegarde un modèle.

    Parameters
    ----------
    model

    filename : str

    Returns
    -------
    Path
    """

    filepath = MODELS_DIR / filename

    joblib.dump(
        model,
        filepath,
    )

    return filepath

def load_model(
    filename: str,
):
    """
    Charge un modèle.

    Parameters
    ----------
    filename : str

    Returns
    -------
    Modèle entraîné.
    """

    filepath = MODELS_DIR / filename

    return joblib.load(
        filepath,
    )


def main():

    dataframe = load_csv_dataset(
        CLEAN_DATASET_FILE,
    )

    data = prepare_dataset(
        dataframe=dataframe,
        feature_names=TOP_3_FEATURES,
        target_column=TARGET_COLUMN,
        scale=False,
    )

    model = train_random_forest(
        data["X_train"],
        data["y_train"],
    )

    filepath = save_model(
        model,
        RANDOM_FOREST_MODEL,
    )

    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(filepath)

    print()

    loaded_model = load_model(
        RANDOM_FOREST_MODEL,
    )

    print("=" * 60)
    print("MODEL LOADED")
    print("=" * 60)

    print(loaded_model)

if __name__ == "__main__":
    main()