"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : pipeline.py

Description :
Préparation des données avant l'entraînement.
des modèles de Machine Learning.

============================================================
"""
# ==========================================================
# Bibliothèques tierces
# ==========================================================
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# Modules du projet
# ==========================================================

from src.features.selection import select_features
from src.features.encoding import encode_target
from src.features.scaling import standardize_features


def prepare_dataset(
    dataframe: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    scale: bool = False,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """
    Prépare les données avant l'entraînement.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset complet.

    feature_names : list[str]
        Variables à sélectionner.

    target_column : str
        Variable cible.

    scale : bool, default=False
        Standardiser les variables.

    test_size : float, default=0.20
        Taille du jeu de test.

    random_state : int, default=42
        Graine aléatoire.

    Returns
    -------
    dict
        Dictionnaire contenant les données préparées.
    """

    # ======================================================
    # Sélection des variables
    # ======================================================

    X = select_features(
        dataframe,
        feature_names,
    )

    # ======================================================
    # Encodage de la cible
    # ======================================================

    y, encoder = encode_target(
        dataframe[target_column],
    )

    # ======================================================
    # Standardisation (optionnelle)
    # ======================================================

    scaler = None

    if scale:

        X, scaler = standardize_features(
            X,
        )

    # ======================================================
    # Séparation Train/Test
    # ======================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # ======================================================
    # Retour
    # ======================================================

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoder": encoder,
        "scaler": scaler,
        "feature_names": feature_names,
        "class_names": encoder.classes_,
    }


def main():

    from src.config import (
        CLEAN_DATASET_FILE,
        TOP_3_FEATURES,
        TARGET_COLUMN,
    )

    from src.data.io import load_csv_dataset

    dataframe = load_csv_dataset(
        CLEAN_DATASET_FILE,
    )

    data = prepare_dataset(
        dataframe=dataframe,
        feature_names=TOP_3_FEATURES,
        target_column=TARGET_COLUMN,
        scale=False,
    )

    print("=" * 60)
    print("PIPELINE")
    print("=" * 60)

    print()

    print("X_train :", data["X_train"].shape)
    print("X_test  :", data["X_test"].shape)

    print()

    print("y_train :", data["y_train"].shape)
    print("y_test  :", data["y_test"].shape)

    print()

    print("Classes :", data["class_names"])

    print()

    print("Variables :", data["feature_names"])

    print()

    print("Scaler :", data["scaler"])


if __name__ == "__main__":
    main()