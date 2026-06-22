"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : download.py

Description :
Télécharge des jeux de données JSON depuis une source distante
et les enregistre dans le dossier data/raw.

Ce module ne réalise aucun nettoyage des données.
Sa responsabilité est uniquement la collecte.

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

import requests

# ==========================================================
# Modules du projet
# ==========================================================

from src.config import HEADERS
from src.utils import ensure_directory

# ==========================================================
# Fonctions
# ==========================================================

def download_json(url: str) -> list[dict]:
    """
    Télécharge des données JSON depuis une URL.

    Parameters
    ----------
    url : str
        Adresse du jeu de données.

    Returns
    -------
    list[dict]
        Données JSON téléchargées.

    Raises
    ------
    requests.HTTPError
        Si le serveur retourne une erreur HTTP.
    """

    # Envoi de la requête HTTP
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60
    )

    # Vérifie que la requête a réussi
    response.raise_for_status()

    # Conversion de la réponse en objet Python
    return response.json()


def save_json(data: list[dict], output_file: Path) -> None:
    """
    Sauvegarde des données JSON de manière sécurisée.

    Les données sont d'abord écrites dans un fichier temporaire.
    Celui-ci remplace ensuite le fichier final uniquement si
    l'écriture s'est déroulée correctement.

    Parameters
    ----------
    data : list[dict]
        Données à sauvegarder.

    output_file : Path
        Fichier de destination.

    Returns
    -------
    None
    """

    # Création du dossier si nécessaire
    ensure_directory(output_file.parent)

    # Fichier temporaire
    temp_file = output_file.with_suffix(".tmp")

    # Écriture des données
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Remplacement atomique
    temp_file.replace(output_file)


# ==========================================================
# Fonctions métier
# ==========================================================

def download_dataset(url: str, output_file: Path) -> None:
    """
    Télécharge un jeu de données JSON puis le sauvegarde.

    Cette fonction orchestre les opérations de téléchargement
    et de sauvegarde, sans connaître les détails de leur
    implémentation.

    Parameters
    ----------
    url : str
        URL du jeu de données.

    output_file : Path
        Chemin du fichier de destination.

    Returns
    -------
    None
    """

    print("=" * 60)
    print("Début du téléchargement...")
    print(f"Source : {url}")

    # Téléchargement des données
    data = download_json(url)

    print(f"{len(data)} objets récupérés.")

    # Sauvegarde
    save_json(data, output_file)

    print(f"Fichier enregistré : {output_file}")
    print("Téléchargement terminé.")
    print("=" * 60)

# ==========================================================
# Point d'entrée du programme
# ==========================================================

def main() -> None:
    """
    Point d'entrée du programme.
    """

    from src.config import (
        CELESTRAK_ACTIVE_URL,
        RAW_DATA_DIR,
    )

    output_file = RAW_DATA_DIR / "active_satellites.json"

    download_dataset(
        url=CELESTRAK_ACTIVE_URL,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()