import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from dimension_predictor.config import (
    load_config
)

from dimension_predictor.preprocessing import (
    clean_text
)


def main():

    config = load_config()

    path = ROOT / config["data"]["raw_texts"]

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("AJOUT DE TEXTES")
    print("=" * 60)

    while True:

        text = input(
            "\nTexte (vide pour terminer) :\n> "
        )

        text = clean_text(text)

        if not text:
            break

        print(
            "\nSource : "
            "1=rapport / "
            "2=entreprise"
        )

        choice = input("> ").strip()

        source = (
            "rapport"
            if choice == "1"
            else "entreprise"
        )

        file_exists = path.exists()

        with path.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            if not file_exists:

                writer.writerow([
                    "id",
                    "Chunk",
                    "Source"
                ])

            # ID simple basé sur le nombre
            # de lignes existantes.
            with path.open(
                "r",
                encoding="utf-8"
            ) as reader:

                row_count = sum(
                    1 for _ in reader
                )

            writer.writerow([
                row_count,
                text,
                source
            ])

        print("Texte enregistré.")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
