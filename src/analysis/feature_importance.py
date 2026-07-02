"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : feature_importance.py

Description :
Calcule l'importance des variables à l'aide
d'un modèle Random Forest.

Responsabilités :
    - Charger le dataset nettoyé
    - Préparer les données
    - Entraîner une Random Forest
    - Calculer l'importance des variables
    - Sauvegarder les résultats
    - Générer le graphique

============================================================
"""

# ==========================================================
# Bibliothèques tierces
# ==========================================================

import matplotlib.pyplot as plt
import pandas as pd

# ==========================================================
# Modules du projet
# ==========================================================

from src.config import (
    CLEAN_DATASET_FILE,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    FEATURE_IMPORTANCE_FILE,
    FEATURE_IMPORTANCE_FIGURE,
)

from src.data.io import load_csv_dataset

from src.features.pipeline import prepare_dataset

from src.models.training import train_random_forest

from src.utils import ensure_directory


def load_clean_dataset() -> pd.DataFrame:
    """
    Charge le dataset nettoyé.

    Returns
    -------
    pd.DataFrame
    """

    return load_csv_dataset(
        CLEAN_DATASET_FILE,
    )


def compute_feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Calcule l'importance des variables.

    Parameters
    ----------
    model
        Modèle entraîné possédant l'attribut
        feature_importances_.

    feature_names : list[str]
        Liste des variables.

    Returns
    -------
    pd.DataFrame
        Tableau des importances.
    """

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    ).reset_index(
        drop=True,
    )

    return importance_df


def save_feature_importance(
    dataframe: pd.DataFrame,
) -> None:
    """
    Sauvegarde le tableau des importances.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    None
    """

    ensure_directory(
        FEATURE_IMPORTANCE_FILE.parent,
    )

    dataframe.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print(
        f"Tableau sauvegardé : "
        f"{FEATURE_IMPORTANCE_FILE}"
    )


def plot_feature_importance(
    dataframe: pd.DataFrame,
) -> None:
    """
    Génère le graphique des importances.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    None
    """

    ensure_directory(
        FEATURE_IMPORTANCE_FIGURE.parent,
    )

    plt.figure(
        figsize=(10, 6),
    )

    plt.barh(
        dataframe["Feature"],
        dataframe["Importance"],
    )

    plt.xlabel(
        "Importance",
    )

    plt.ylabel(
        "Variables",
    )

    plt.title(
        "Feature Importance (Random Forest)",
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        FEATURE_IMPORTANCE_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Figure sauvegardée : "
        f"{FEATURE_IMPORTANCE_FIGURE}"
    )

    plt.close()


def main() -> None:
    """
    Test du module.
    """

    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    dataframe = load_clean_dataset()

    data = prepare_dataset(
        dataframe=dataframe,
        feature_names=NUMERICAL_FEATURES,
        target_column=TARGET_COLUMN,
        scale=False,
    )

    print()

    print("Train/Test Split")

    print("-" * 60)

    print(f"X_train : {data['X_train'].shape}")
    print(f"X_test  : {data['X_test'].shape}")
    print(f"y_train : {data['y_train'].shape}")
    print(f"y_test  : {data['y_test'].shape}")

    model = train_random_forest(
        data["X_train"],
        data["y_train"],
    )

    importance_df = compute_feature_importance(
        model,
        data["feature_names"],
    )

    print()

    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print()

    print(importance_df)

    print()

    save_feature_importance(
        importance_df,
    )

    plot_feature_importance(
        importance_df,
    )

    print()

    print("=" * 60)
    print("MODEL")
    print("=" * 60)

    print(model)


if __name__ == "__main__":
    main()