"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : io.py

Description :
Fonctions d'entrée/sortie (Input / Output) utilisées
dans l'ensemble du projet.

Responsabilités :
    - Charger des fichiers JSON
    - Charger des fichiers CSV
    - Sauvegarder des fichiers JSON
    - Sauvegarder des fichiers CSV

============================================================
"""

# ==========================================================
# Bibliothèques standard
# ==========================================================

from pathlib import Path
import json

# ==========================================================
# Bibliothèques tierces
# ==========================================================

import pandas as pd

# ==========================================================
# Modules du projet
# ==========================================================

from src.utils import ensure_directory

def load_json_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Charge un fichier JSON dans un DataFrame.

    Parameters
    ----------
    file_path : Path
        Chemin du fichier JSON.

    Returns
    -------
    pd.DataFrame
        Données chargées.

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
            f"Fichier introuvable : {file_path}"
        )

    # Vérifie que le fichier n'est pas vide
    if file_path.stat().st_size == 0:
        raise ValueError(
            f"Le fichier '{file_path.name}' est vide."
        )
    

    # Chargement du DataFrame
    try:
        return pd.read_json(file_path)
    except ValueError as error:
        raise ValueError(
            f"Le fichier '{file_path.name}' n'est pas un JSON valide."
        ) from error

    
def load_csv_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Charge un fichier CSV dans un DataFrame.

    Parameters
    ----------
    file_path : Path
        Chemin du fichier CSV.

    Returns
    -------
    pd.DataFrame
        Données chargées.

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
            f"Fichier introuvable : {file_path}"
        )

    # Vérifie que le fichier n'est pas vide
    if file_path.stat().st_size == 0:
        raise ValueError(
            f"Le fichier '{file_path.name}' est vide."
        )

    try:
        return pd.read_csv(file_path)

    except Exception as error:
        raise ValueError(
            f"Impossible de lire le fichier CSV '{file_path.name}'."
        ) from error
    

def save_csv_dataset(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Sauvegarde un DataFrame au format CSV.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Données à sauvegarder.

    output_file : Path
        Fichier de destination.

    Returns
    -------
    None
    """

    # Création du dossier si nécessaire
    ensure_directory(output_file.parent)

    # Sauvegarde
    dataframe.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print(f"Dataset sauvegardé : {output_file}")