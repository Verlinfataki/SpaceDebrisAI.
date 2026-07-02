from src.config import (
    CLEAN_DATASET_FILE,
    TOP_3_FEATURES,
)

from src.data.io import load_csv_dataset
import pandas as pd


def select_features(
    dataframe: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Sélectionne un ensemble de variables.

    Parameters
    ----------
    dataframe : pd.DataFrame

    features : list[str]

    Returns
    -------
    pd.DataFrame
    """

    return dataframe[
        features
    ].copy()


def main() -> None:

    dataframe = load_csv_dataset(
        CLEAN_DATASET_FILE,
    )

    selected = select_features(
        dataframe,
        TOP_3_FEATURES,
    )

    print("=" * 60)
    print("TOP 3 FEATURES")
    print("=" * 60)

    print(selected.head())

    print()

    print(selected.shape)


if __name__ == "__main__":
    main()