"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : training.py

Description :
Entraîne les modèles de Machine Learning.

Responsabilités :
    - Construire les modèles
    - Les entraîner
    - Retourner le modèle entraîné

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

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.data.io import load_csv_dataset
from src.features.pipeline import prepare_dataset



def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Entraîne un modèle Random Forest.

    Parameters
    ----------
    X_train : pd.DataFrame
        Variables d'entraînement.

    y_train : pd.Series
        Variable cible.

    n_estimators : int
        Nombre d'arbres.

    random_state : int
        Graine aléatoire.

    Returns
    -------
    RandomForestClassifier
        Modèle entraîné.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


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

    print("=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    print(model)

if __name__ == "__main__":
    main()