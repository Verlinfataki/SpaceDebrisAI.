"""
============================================================
Projet : SpaceDebrisAI
Auteur : Verlin Fataki
Fichier : encoding.py

Description :
Fonctions d'encodage des variables.

Responsabilités :
    - Encoder la variable cible
    - Décoder la variable cible

============================================================
"""
# ==========================================================
# Bibliothèques tierces
# ==========================================================

import pandas as pd

from sklearn.preprocessing import LabelEncoder

def encode_target(
    target: pd.Series,
) -> tuple[pd.Series, LabelEncoder]:
    """
    Encode la variable cible.

    Parameters
    ----------
    target : pd.Series
        Variable cible.

    Returns
    -------
    tuple
        Variable encodée et encodeur.
    """

    encoder = LabelEncoder()

    encoded_target = encoder.fit_transform(
        target
    )

    return (
        pd.Series(
            encoded_target,
            index=target.index,
            name=target.name,
        ),
        encoder,
    )


def decode_target(
    target: pd.Series,
    encoder: LabelEncoder,
) -> pd.Series:
    """
    Décode la variable cible.

    Parameters
    ----------
    target : pd.Series
        Variable encodée.

    encoder : LabelEncoder
        Encodeur entraîné.

    Returns
    -------
    pd.Series
    """

    decoded = encoder.inverse_transform(
        target
    )

    return pd.Series(
        decoded,
        index=target.index,
    )


def main() -> None:

    target = pd.Series(
        [
            "Satellite",
            "Debris",
            "Satellite",
            "Debris",
        ]
    )

    encoded, encoder = encode_target(
        target
    )

    print("Original")
    print(target)

    print()

    print("Encodé")
    print(encoded)

    print()

    print("Classes")
    print(encoder.classes_)

    print()

    decoded = decode_target(
        encoded,
        encoder,
    )

    print("Décodé")
    print(decoded)


if __name__ == "__main__":
    main()