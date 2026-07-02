"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : prediction.py

Description :
Effectue les prédictions à partir
d'un modèle entraîné.

Responsabilités :
    - Prédire les classes
    - Prédire les probabilités

============================================================
"""

# ==========================================================
# Bibliothèques tierces
# ==========================================================
from src.config import (
    CLEAN_DATASET_FILE,
    TARGET_COLUMN,
    TOP_3_FEATURES,
)

from src.data.io import load_csv_dataset
from src.features.pipeline import prepare_dataset
from src.models.training import train_random_forest
import numpy as np
import pandas as pd

def predict(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Effectue des prédictions.

    Parameters
    ----------
    model
        Modèle entraîné.

    X : pd.DataFrame
        Variables explicatives.

    Returns
    -------
    np.ndarray
        Classes prédites.
    """

    return model.predict(X)


def predict_probabilities(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Retourne les probabilités
    de chaque classe.

    Parameters
    ----------
    model
        Modèle entraîné.

    X : pd.DataFrame

    Returns
    -------
    np.ndarray
    """

    return model.predict_proba(X)


def decode_predictions(
    predictions: np.ndarray,
    encoder,
) -> pd.Series:
    """
    Décode les classes prédites.

    Parameters
    ----------
    predictions : np.ndarray
        Classes prédites.

    encoder : LabelEncoder

    Returns
    -------
    pd.Series
    """

    decoded = encoder.inverse_transform(
        predictions
    )

    return pd.Series(
        decoded,
        name="Predictions",
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

    predictions = predict(
        model,
        data["X_test"],
    )

    print("=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    print(predictions[:10])

    print()

    probabilities = predict_probabilities(
        model,
        data["X_test"],
    )

    print("Probabilités")

    print(probabilities[:5])

    decoded = decode_predictions(
        predictions,
        data["encoder"],
    )

    print()

    print(decoded.head(10))

if __name__ == "__main__":
    main()