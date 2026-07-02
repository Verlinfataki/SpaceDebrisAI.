"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : feature_importance.py

Description :
Calcule l'importance des variables à l'aide
d'une Random Forest.

Responsabilités :
    - Charger le dataset nettoyé
    - Préparer X et y
    - Séparer Train/Test
    - Entraîner une Random Forest
    - Calculer l'importance des variables
    - Afficher les résultats

============================================================
"""
# ==========================================================
# Bibliothèques tierces
# ==========================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
import matplotlib.pyplot as plt
from src.utils import ensure_directory

def load_clean_dataset() -> pd.DataFrame:
    """
    Charge le dataset nettoyé.
    """

    return load_csv_dataset(
        CLEAN_DATASET_FILE,
    )


def prepare_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prépare X et y.

    Returns
    -------
    tuple
        X, y
    """

    X = dataframe[
        NUMERICAL_FEATURES
    ]

    y = dataframe[
        TARGET_COLUMN
    ]

    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Sépare les données en ensembles
    d'entraînement et de test.

    Parameters
    ----------
    X : pd.DataFrame
        Variables explicatives.

    y : pd.Series
        Variable cible.

    Returns
    -------
    tuple
        X_train, X_test,
        y_train, y_test
    """

    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def compute_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Calcule l'importance des variables.

    Parameters
    ----------
    model : RandomForestClassifier
        Modèle entraîné.

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
    ).reset_index(drop=True)

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
        FEATURE_IMPORTANCE_FILE.parent
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
    Affiche l'importance des variables.
    """

    ensure_directory(
        FEATURE_IMPORTANCE_FIGURE.parent
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        dataframe["Feature"],
        dataframe["Importance"],
    )

    plt.xlabel("Importance")

    plt.ylabel("Variables")

    plt.title(
        "Importance des variables"
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

    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    dataframe = load_clean_dataset()

    X, y = prepare_dataset(
        dataframe,
    )

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    print()
    print("Train/Test Split")
    print("-" * 60)
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")

    model = train_random_forest(
        X_train,
        y_train,
    )

    importance_df = compute_feature_importance(
        model,
        NUMERICAL_FEATURES,
    )

    print()
    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(importance_df)

    save_feature_importance(
    importance_df,
    )

    plot_feature_importance(
        importance_df,
    )

    print()
    print("=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)
    print(model)


if __name__ == "__main__":
    main()