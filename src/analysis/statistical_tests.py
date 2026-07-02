"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : statistical_tests.py

Description :
Effectue les tests statistiques entre les satellites
et les débris.

Responsabilités :
    - Charger le dataset nettoyé
    - Effectuer le test de Mann-Whitney U
    - Interpréter les résultats

============================================================
"""

# ==========================================================
# Bibliothèques tierces
# ==========================================================

import pandas as pd

from scipy.stats import mannwhitneyu

# ==========================================================
# Modules du projet
# ==========================================================

from src.config import (
    CLEAN_DATASET_FILE,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    SIGNIFICANCE_LEVEL,
)

from src.data.io import load_csv_dataset


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



def mann_whitney_test(
    dataframe: pd.DataFrame,
    column: str,
) -> tuple[float, float]:
    """
    Effectue le test de Mann-Whitney U.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset.

    column : str
        Variable à tester.

    Returns
    -------
    tuple[float, float]
        Statistique U et p-value.
    """

    satellites = dataframe.loc[
        dataframe[TARGET_COLUMN] == "Satellite",
        column,
    ]

    debris = dataframe.loc[
        dataframe[TARGET_COLUMN] == "Debris",
        column,
    ]

    statistic, p_value = mannwhitneyu(
        satellites,
        debris,
        alternative="two-sided",
    )

    return statistic, p_value



def compute_rank_biserial(
    statistic: float,
    n1: int,
    n2: int,
) -> float:
    """
    Calcule la corrélation bisérielle de rang
    (Rank-Biserial Correlation).

    Parameters
    ----------
    statistic : float
        Statistique U.

    n1 : int
        Taille du premier groupe.

    n2 : int
        Taille du second groupe.

    Returns
    -------
    float
        Taille d'effet.
    """

    return (2 * statistic) / (n1 * n2) - 1



def interpret_p_value(
    p_value: float,
    alpha: float = 0.05,
) -> str:
    """
    Interprète la p-value du test statistique.

    Parameters
    ----------
    p_value : float
        p-value obtenue.

    alpha : float, default=0.05
        Niveau de significativité.

    Returns
    -------
    str
        Interprétation.
    """

    if p_value < alpha:
        return (
            "✔ Rejet de H₀ : différence "
            "statistiquement significative."
        )

    return (
        "✘ On ne rejette pas H₀ : "
        "aucune différence significative."
    )


def interpret_effect_size(
    effect_size: float,
) -> str:
    """
    Interprète la taille d'effet.

    Parameters
    ----------
    effect_size : float

    Returns
    -------
    str
    """

    value = abs(effect_size)

    if value < 0.10:
        return "Effet négligeable"

    if value < 0.30:
        return "Petit effet"

    if value < 0.50:
        return "Effet moyen"

    return "Grand effet"



def analyze_feature(
    dataframe: pd.DataFrame,
    column: str,
) -> dict:
    """
    Analyse statistiquement une variable.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset nettoyé.

    column : str
        Variable à analyser.

    Returns
    -------
    dict
        Résultats du test.
    """

    print("=" * 60)
    print(f"Analyse : {column}")
    print("=" * 60)

    statistic, p_value = mann_whitney_test(
        dataframe,
        column,
    )

    satellites = dataframe.loc[
        dataframe[TARGET_COLUMN] == "Satellite",
        column,
    ]

    debris = dataframe.loc[
        dataframe[TARGET_COLUMN] == "Debris",
        column,
    ]

    effect_size = compute_rank_biserial(
        statistic,
        len(satellites),
        len(debris),
    )

    decision = interpret_p_value(
        p_value,
        SIGNIFICANCE_LEVEL,
    )

    effect = interpret_effect_size(
        effect_size,
    )

    print(f"Statistique U : {statistic:.2f}")
    print(f"p-value       : {p_value:.6f}")
    print(f"Taille effet  : {effect_size:.4f}")

    print()
    print(decision)
    print(effect)

    return {
        "Variable": column,
        "Statistic_U": statistic,
        "P_Value": p_value,
        "Effect_Size": effect_size,
        "Decision": decision,
        "Effect": effect,
    }



def main() -> None:

    print("=" * 60)
    print("TESTS STATISTIQUES")
    print("=" * 60)

    dataframe = load_clean_dataset()

    results = []

    for column in NUMERICAL_FEATURES:

        result = analyze_feature(
            dataframe,
            column,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print()
    print(results_df)


if __name__ == "__main__":
    main()