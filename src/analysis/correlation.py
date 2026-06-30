"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : correlation.py

Description :
Analyse des corrélations entre les variables numériques.

Responsabilités :
    - Charger le dataset nettoyé
    - Calculer la matrice de corrélation
    - Générer une heatmap
    - Sauvegarder la figure

============================================================
"""

# ==========================================================
# Bibliothèques tierces
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Modules du projet
# ==========================================================

from src.config import (
    CLEAN_DATASET_FILE,
    NUMERICAL_FEATURES,
    FIGURES_DIR,
)

from src.data.io import load_csv_dataset

from src.utils import ensure_directory
from src.visualization.plotting import save_figure


def load_clean_dataset() -> pd.DataFrame:
    """
    Charge le dataset nettoyé.
    """

    return load_csv_dataset(
        CLEAN_DATASET_FILE,
    )


def compute_correlation_matrix(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    return dataframe[
        NUMERICAL_FEATURES
    ].corr()

def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
) -> None:
    """
    Affiche et sauvegarde la matrice de corrélation.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Matrice de corrélation.

    Returns
    -------
    None
    """

    plt.figure(figsize=(10, 8))

    image = plt.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(image)

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90,
    )

    plt.yticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
    )

    # Affiche les valeurs dans chaque cellule
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):

            plt.text(
                j,
                i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.title("Matrice de corrélation")

    save_figure(
        "correlation_matrix.png",
    )

def main() -> None:

    dataframe = load_clean_dataset()

    correlation = compute_correlation_matrix(
        dataframe,
    )

    print(correlation)

    plot_correlation_heatmap(
        correlation,
    )


if __name__ == "__main__":
    main()