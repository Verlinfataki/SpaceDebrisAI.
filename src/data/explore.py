"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : explore.py

Description :
Charge un jeu de données et fournit des outils
d'exploration (EDA - Exploratory Data Analysis).

Ce module ne modifie jamais les données.
Il les analyse uniquement.

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

from src.config import RAW_DATA_DIR

# ==========================================================
# Fonctions
# ==========================================================
from src.utils import print_section




def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Charge un jeu de données au format JSON.

    Parameters
    ----------
    file_path : Path
        Chemin vers le fichier JSON.

    Returns
    -------
    pd.DataFrame
        Les données chargées sous forme de DataFrame Pandas.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.

    ValueError
        Si le fichier est vide.
    """

    # Vérifie que le fichier existe
    if not file_path.exists():
        raise FileNotFoundError(
            f"Le fichier est introuvable : {file_path}"
        )

    # Chargement des données
    dataframe = pd.read_json(file_path)

    # Vérifie que le DataFrame n'est pas vide
    if dataframe.empty:
        raise ValueError(
            f"Le fichier est vide : {file_path}"
        )

    return dataframe


# ==========================================================
# Fonctions d'exploration
# ==========================================================

def dataset_summary(dataframe: pd.DataFrame) -> None:
    """
    Affiche un résumé général du jeu de données.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Jeu de données chargé.

    Returns
    -------
    None
    """

    # ------------------------------------------------------
    # Dimensions du dataset
    # ------------------------------------------------------
    print_section("Dataset Summary")

    rows, columns = dataframe.shape

    print(f"[INFO] Nombre de lignes   : {rows:,}")
    print(f"[INFO] Nombre de colonnes : {columns}")

    # ------------------------------------------------------
    # Aperçu des données
    # ------------------------------------------------------
    print_section("Aperçu du dataset")

    print(dataframe.head())

    # ------------------------------------------------------
    # Informations générales
    # ------------------------------------------------------
    print_section("Informations générales")

    dataframe.info()

    # ------------------------------------------------------
    # Colonnes disponibles
    # ------------------------------------------------------
    print_section("Colonnes")

    for index, column in enumerate(dataframe.columns, start=1):
        print(f"{index:2d}. {column}")

    # ------------------------------------------------------
    # Statistiques descriptives
    # ------------------------------------------------------
    print_section("Statistiques descriptives")

    print(dataframe.describe())


# ==========================================================
# Point d'entrée
# ==========================================================

def main() -> None:
    """
    Point d'entrée du module d'exploration.
    """

    dataset_path = RAW_DATA_DIR / "active_satellites.json"

    dataframe = load_dataset(dataset_path)

    dataset_summary(dataframe)


if __name__ == "__main__":
    main()