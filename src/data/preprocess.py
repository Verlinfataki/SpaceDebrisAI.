"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : preprocess.py

Description :
Prépare le dataset d'entraînement avant la phase de
Machine Learning.

Responsabilités :
    - Charger le dataset d'entraînement
    - Vérifier la qualité des données
    - Produire un rapport de qualité
    - Sauvegarder le dataset nettoyé

============================================================
"""

# ==========================================================
# Bibliothèques standard
# ==========================================================

from pathlib import Path

# ==========================================================
# Bibliothèques tierces
# ==========================================================

import pandas as pd

# ==========================================================
# Modules du projet
# ==========================================================

from src.config import (
    TRAINING_DATASET_FILE,
    CLEAN_DATASET_FILE,
)

from src.data.io import (
    load_csv_dataset,
    save_csv_dataset,
)

def load_training_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Charge le dataset d'entraînement.

    Parameters
    ----------
    file_path : Path
        Chemin du fichier CSV.

    Returns
    -------
    pd.DataFrame
        Dataset d'entraînement.
    """

    return load_csv_dataset(file_path)



def get_dataset_summary(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """
    Retourne les informations générales du dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset à analyser.

    Returns
    -------
    dict[str, object]
        Résumé du dataset.
    """

    summary = {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "column_names": dataframe.columns.tolist(),
        "dtypes": dataframe.dtypes.astype(str).to_dict(),
        "memory_usage_mb": round(
            dataframe.memory_usage(deep=True).sum() / (1024 ** 2),
            2,
        ),
    }

    return summary


def check_missing_values(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Retourne le nombre de valeurs manquantes
    pour chaque colonne.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset à analyser.

    Returns
    -------
    pd.Series
        Nombre de valeurs manquantes.
    """

    return dataframe.isna().sum()


def check_duplicates(
    dataframe: pd.DataFrame,
) -> int:
    """
    Retourne le nombre de lignes dupliquées.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset à analyser.

    Returns
    -------
    int
        Nombre de doublons.
    """

    return int(dataframe.duplicated().sum())


def check_class_distribution(
    dataframe: pd.DataFrame,
) -> pd.Series:
    
    return dataframe["LABEL"].value_counts().sort_index()


def check_column_types(
    dataframe: pd.DataFrame,
) -> dict[str, str]:
    """
    Retourne les types des colonnes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset à analyser.

    Returns
    -------
    dict[str, str]
        Types des colonnes.
    """

    return dataframe.dtypes.astype(str).to_dict()


def save_clean_dataset(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Sauvegarde le dataset nettoyé.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset à sauvegarder.

    output_file : Path
        Chemin du fichier de destination.

    Returns
    -------
    None
    """

    save_csv_dataset(
        dataframe,
        output_file,
    )


# ==========================================================
# Point d'entrée du programme
# ==========================================================

def main() -> None:
    """
    Point d'entrée du prétraitement.
    """

    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    # Chargement
    dataframe = load_training_dataset(
        TRAINING_DATASET_FILE,
    )

    # Résumé général
    summary = get_dataset_summary(dataframe)

    print(f"Lignes    : {summary['rows']:,}")
    print(f"Colonnes  : {summary['columns']}")
    print(
        f"Mémoire   : "
        f"{summary['memory_usage_mb']} MB"
    )

    # Valeurs manquantes
    print("\nValeurs manquantes")
    print("-" * 60)
    print(check_missing_values(dataframe))

    # Doublons
    print("\nDoublons")
    print("-" * 60)
    print(check_duplicates(dataframe))

    # Répartition des classes
    print("\nRépartition des classes")
    print("-" * 60)
    print(check_class_distribution(dataframe))

    # Types
    print("\nTypes des colonnes")
    print("-" * 60)

    for column, dtype in (
        check_column_types(dataframe).items()
    ):
        print(f"{column}: {dtype}")

    # Sauvegarde
    save_clean_dataset(
        dataframe,
        CLEAN_DATASET_FILE,
    )

    print("=" * 60)
    print("Prétraitement terminé.")
    print("=" * 60)


if __name__ == "__main__":
    main()


