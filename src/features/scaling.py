"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : scaling.py

Description :
Fonctions de mise à l'échelle des variables.

Responsabilités :
    - Standardiser les variables
    - Normaliser les variables
    - Réutiliser les scalers

============================================================
"""

# ==========================================================
# Bibliothèques tierces
# ==========================================================

import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)

def standardize_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise les variables.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    tuple
        DataFrame standardisé
        et scaler entraîné.
    """

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        dataframe
    )

    scaled_dataframe = pd.DataFrame(
        scaled,
        columns=dataframe.columns,
        index=dataframe.index,
    )

    return (
        scaled_dataframe,
        scaler,
    )


def normalize_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalise les variables.

    Returns
    -------
    tuple
    """

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        dataframe
    )

    scaled_dataframe = pd.DataFrame(
        scaled,
        columns=dataframe.columns,
        index=dataframe.index,
    )

    return (
        scaled_dataframe,
        scaler,
    )


def main() -> None:

    from src.config import (
        CLEAN_DATASET_FILE,
        NUMERICAL_FEATURES,
    )

    from src.data.io import load_csv_dataset

    dataframe = load_csv_dataset(
        CLEAN_DATASET_FILE,
    )

    features = dataframe[
        NUMERICAL_FEATURES
    ]

    print("=" * 60)
    print("STANDARD SCALER")
    print("=" * 60)

    standardized, standard_scaler = standardize_features(
        features,
    )

    print(standardized.head())

    print()

    print(standardized.describe())

    print()

    print("=" * 60)
    print("MINMAX SCALER")
    print("=" * 60)

    normalized, minmax_scaler = normalize_features(
        features,
    )

    print(normalized.head())

    print()

    print(normalized.describe())


if __name__ == "__main__":
    main()