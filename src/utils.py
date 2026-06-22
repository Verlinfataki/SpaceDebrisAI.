from pathlib import Path


def ensure_directory(directory: Path):
    """
    Crée un dossier s'il n'existe pas.
    """
    directory.mkdir(parents=True, exist_ok=True)