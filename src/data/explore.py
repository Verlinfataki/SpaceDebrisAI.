import json
from pathlib import Path

DATA_FILE = Path("data/raw/active_satellites.json")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 50)
print(f"Nombre d'objets : {len(data)}")
print("=" * 50)

print("\nPremier objet :\n")
for key, value in data[0].items():
    print(f"{key:25} : {value}")