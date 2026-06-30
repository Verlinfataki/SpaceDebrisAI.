"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : merge.py

Description :
Fusionne plusieurs jeux de données en un unique dataset
d'entraînement.

Chaque dataset reçoit une colonne LABEL avant la fusion.

Responsabilités :
    - Charger les datasets
    - Ajouter les labels
    - Fusionner les données
    - Sauvegarder le dataset final

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
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATASET_SOURCES,
)

from src.data.io import (
    load_json_dataset,
    save_csv_dataset,
)

# ==========================================================
# Fonctions
# ==========================================================

def load_dataset_with_label(
    file_path: Path,
    label: str,
) -> pd.DataFrame:
    """
    Charge un dataset JSON et ajoute une colonne LABEL.

    Parameters
    ----------
    file_path : Path
        Chemin du fichier JSON.

    label : str
        Classe à associer au dataset.

    Returns
    -------
    pd.DataFrame
        Dataset chargé avec la colonne LABEL.
    """

    # Chargement du dataset
    dataframe = load_json_dataset(file_path)

    # Ajout du label
    dataframe["LABEL"] = label

    return dataframe


def merge_datasets(
    datasets: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Fusionne plusieurs DataFrames.

    Parameters
    ----------
    datasets : list[pd.DataFrame]
        Liste des DataFrames.

    Returns
    -------
    pd.DataFrame
        Dataset fusionné.
    """

    merged_dataframe = pd.concat(
        datasets,
        ignore_index=True,
    )

    return merged_dataframe


# ==========================================================
# Point d'entrée
# ==========================================================

def main() -> None:
    """
    Construit le dataset d'entraînement.

    Returns
    -------
    None
    """

    print("=" * 60)
    print("Construction du dataset d'entraînement...")
    print("=" * 60)

    datasets: list[pd.DataFrame] = []

    # Chargement des jeux de données
    for source in DATASET_SOURCES:

        file_path = RAW_DATA_DIR / source["filename"]

        print(f"Lecture : {file_path.name}")

        dataframe = load_dataset_with_label(
            file_path=file_path,
            label=source["label"],
        )

        print(
            f"   {len(dataframe):,} lignes "
            f"({source['label']})"
        )

        datasets.append(dataframe)

    # Fusion
    dataframe = merge_datasets(datasets)

    print("-" * 60)
    print(f"Nombre total de lignes : {len(dataframe):,}")

    # Répartition des classes
    print("\nRépartition des classes")
    print(dataframe["LABEL"].value_counts())

    # Sauvegarde
    output_file = (
        PROCESSED_DATA_DIR /
        "training_dataset.csv"
    )

    save_csv_dataset(
        dataframe,
        output_file,
    )

    print("=" * 60)
    print("Fusion terminée.")
    print("=" * 60)


if __name__ == "__main__":
    main()