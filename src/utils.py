"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : utils.py

Description :
Fonctions utilitaires utilisées dans l'ensemble du projet.

Ces fonctions ne sont liées à aucun domaine métier
particulier (téléchargement, IA, visualisation, etc.).

============================================================
"""

# ==========================================================
# Bibliothèques standard
# ==========================================================

from pathlib import Path


# ==========================================================
# Fonctions utilitaires
# ==========================================================

def ensure_directory(directory: Path) -> None:
    """
    Crée un dossier s'il n'existe pas.

    Parameters
    ----------
    directory : Path
        Dossier à créer.

    Returns
    -------
    None
    """

    directory.mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    """
    Affiche un titre de section formaté dans le terminal.

    Parameters
    ----------
    title : str
        Titre à afficher.

    Returns
    -------
    None
    """

    separator = "=" * 60

    print()
    print(separator)
    print(title.upper())
    print(separator)    