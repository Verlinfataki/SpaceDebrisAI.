"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : feature_analysis.py

Description :
Analyse scientifique des variables du dataset.

Responsabilités :
    - Charger le dataset nettoyé
    - Produire des statistiques descriptives
    - Visualiser les distributions
    - Comparer les variables selon les classes
    - Sauvegarder les figures

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
import matplotlib.pyplot as plt

# ==========================================================
# Modules du projet
# ==========================================================
from src.config import (
    CLEAN_DATASET_FILE,
    FIGURES_DIR,
    NUMERICAL_FEATURES,
     TARGET_COLUMN,
)

from src.data.io import load_csv_dataset
from src.utils import ensure_directory
from src.visualization.plotting import save_figure


def load_clean_dataset() -> pd.DataFrame:
    """
    Charge le dataset nettoyé.

    Returns
    -------
    pd.DataFrame
        Dataset nettoyé.
    """

    return load_csv_dataset(
        CLEAN_DATASET_FILE,
    )


def get_descriptive_statistics(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Retourne les statistiques descriptives
    d'une variable numérique.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset à analyser.

    column : str
        Nom de la colonne.

    Returns
    -------
    pd.Series
        Statistiques descriptives.
    """

    return dataframe[column].describe()



def plot_histogram(
    dataframe: pd.DataFrame,
    column: str,
    bins: int = 30,
) -> None:
    """
    Affiche et sauvegarde l'histogramme d'une variable
    selon les classes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset à analyser.

    column : str
        Nom de la variable.

    bins : int, default=30
        Nombre de classes.

    Returns
    -------
    None
    """

    # Création du dossier si nécessaire
    ensure_directory(FIGURES_DIR)

    plt.figure(figsize=(10, 6))

    # Couleurs des classes
    colors = {
        "Satellite": "blue",
        "Debris": "red",
    }

    # Histogrammes
    for label, color in colors.items():

        plt.hist(
            dataframe.loc[
                dataframe[TARGET_COLUMN] == label,
                column,
            ],
            bins=bins,
            alpha=0.6,
            label=label,
            color=color,
        )

    plt.title(f"Distribution de {column}")

    plt.xlabel(column)

    plt.ylabel("Nombre d'observations")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    # Sauvegarde
    save_figure(
        f"{column.lower()}_histogram.png",
    )


def plot_boxplot(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    """
    Affiche un boxplot par classe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset à analyser.

    column : str
        Variable numérique.

    Returns
    -------
    None
    """

    plt.figure(figsize=(10, 6))

    dataframe.boxplot(
        column=column,
        by=TARGET_COLUMN,
    )

    plt.title(f"Boxplot de {column}")

    plt.suptitle("")

    plt.xlabel("Classe")

    plt.ylabel(column)

    plt.grid(True)

    save_figure(
        f"{column.lower()}_boxplot.png",
    )



def analyze_feature(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    """
    Analyse complète d'une variable.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset à analyser.

    column : str
        Variable numérique.

    Returns
    -------
    None
    """

    print("=" * 60)
    print(f"Analyse de : {column}")
    print("=" * 60)

    print("\nStatistiques descriptives")
    print("-" * 60)

    print(
        get_descriptive_statistics(
            dataframe,
            column,
        )
    )

    print()

    plot_histogram(
        dataframe,
        column,
    )

    plot_boxplot(
        dataframe,
        column,
    )


def main() -> None:
    """
    Point d'entrée de l'analyse des variables.
    """

    print("=" * 60)
    print("ANALYSE DES VARIABLES")
    print("=" * 60)

    dataframe = load_clean_dataset()

    for column in NUMERICAL_FEATURES:

        analyze_feature(
            dataframe,
            column,
        )

    print()
    print("=" * 60)
    print("Analyse terminée.")
    print("=" * 60)


if __name__ == "__main__":
    main()