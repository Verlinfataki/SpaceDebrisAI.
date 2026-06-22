import json
import requests
from pathlib import Path

URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "active_satellites.json"

def download_data():
    print("Téléchargement...")

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    # Vérifie que le JSON est valide
    data = response.json()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"{len(data)} objets téléchargés.")
    print(f"Fichier enregistré : {OUTPUT_FILE}")

if __name__ == "__main__":
    download_data()