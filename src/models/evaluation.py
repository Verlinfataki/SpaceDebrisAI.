"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : evaluation.py

Description :
Évalue les performances d'un modèle
de Machine Learning.

Responsabilités :
    - Calculer les métriques
    - Construire la matrice de confusion
    - Générer le rapport de classification

============================================================
"""

# ==========================================================
# Bibliothèques tierces
# ==========================================================

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ==========================================================
# Modules du projet
# ==========================================================

from src.config import (
    CLEAN_DATASET_FILE,
    TARGET_COLUMN,
    TOP_3_FEATURES,
)

from src.data.io import load_csv_dataset
from src.features.pipeline import prepare_dataset
from src.models.prediction import predict
from src.models.training import train_random_forest


def evaluate_model(
    y_true,
    y_pred,
) -> dict:
    """
    Évalue les performances d'un modèle
    de classification.

    Parameters
    ----------
    y_true : pd.Series
        Valeurs réelles.

    y_pred : pd.Series
        Valeurs prédites.

    Returns
    -------
    dict
        Résultats de l'évaluation.
    """

    # ======================================================
    # Métriques globales
    # ======================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
    )

    recall = recall_score(
        y_true,
        y_pred,
    )

    f1 = f1_score(
        y_true,
        y_pred,
    )

    # ======================================================
    # Matrice de confusion
    # ======================================================

    confusion_df = pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
        ),
        index=[
            "Actual Debris",
            "Actual Satellite",
        ],
        columns=[
            "Predicted Debris",
            "Predicted Satellite",
        ],
    )

    # ======================================================
    # Rapport de classification
    # ======================================================

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    # ======================================================
    # Retour
    # ======================================================

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "confusion_matrix": confusion_df,

        "classification_report": report_df,

        "n_samples": len(y_true),
    }


def main() -> None:
    """
    Test du module.
    """

    dataframe = load_csv_dataset(
        CLEAN_DATASET_FILE,
    )

    data = prepare_dataset(
        dataframe=dataframe,
        feature_names=TOP_3_FEATURES,
        target_column=TARGET_COLUMN,
        scale=False,
    )

    model = train_random_forest(
        data["X_train"],
        data["y_train"],
    )

    predictions = predict(
        model,
        data["X_test"],
    )

    results = evaluate_model(
        data["y_test"],
        predictions,
    )

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print()

    print(f"Number of samples : {results['n_samples']}")

    print()

    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1-score  : {results['f1_score']:.4f}")

    print()

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(results["confusion_matrix"])

    print()

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(results["classification_report"])


if __name__ == "__main__":
    main()