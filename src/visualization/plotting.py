"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : plotting.py

Description :
Fonctions utilitaires pour la visualisation.

============================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import FIGURES_DIR
from src.utils import ensure_directory


def save_figure(
    filename: str,
) -> None:
    """
    Sauvegarde la figure courante.

    Parameters
    ----------
    filename : str
        Nom du fichier image.

    Returns
    -------
    None
    """

    ensure_directory(FIGURES_DIR)

    output_file = FIGURES_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Figure sauvegardée : {output_file}")

    plt.show()

    plt.close()