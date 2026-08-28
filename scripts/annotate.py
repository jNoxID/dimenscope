import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from dimension_predictor.config import (
    load_config,
    load_labels
)

from dimension_predictor.labels import (
    get_allowed_statuses
)


def main():

    config = load_config()
    labels = load_labels()

    raw_path = (
        ROOT
        / config["data"]["raw_texts"]
    )

    corpus_path = (
        ROOT
        / config["data"]["corpus"]
    )

    if not raw_path.exists():

        raise FileNotFoundError(
            "Aucun texte brut trouvé. "
            "Lancez d'abord : "
            "python scripts/add_text.py"
        )

    raw_df = pd.read_csv(
        raw_path
    )

    corpus_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if corpus_path.exists():

        corpus_df = pd.read_csv(
            corpus_path
        )

        annotated_ids = set(
            corpus_df["id"].tolist()
        )

    else:

        annotated_ids = set()

    columns = [
        "id",
        "Chunk",
        "Source"
    ] + [
        dimension["id"]
        for dimension
        in labels["dimensions"]
    ]

    for _, row in raw_df.iterrows():

        row_id = row["id"]

        if row_id in annotated_ids:
            continue

        print()
        print("=" * 70)
        print(
            f"ANNOTATION DU TEXTE {row_id}"
        )
        print("=" * 70)

        print()
        print(row["Chunk"])

        print()
        print(
            "Source :",
            row["Source"]
        )

        annotation = {
            "id": row_id,
            "Chunk": row["Chunk"],
            "Source": row["Source"],
        }

        for dimension in labels[
            "dimensions"
        ]:

            allowed = (
                get_allowed_statuses(
                    labels,
                    dimension,
                    row["Source"]
                )
            )

            print()
            print(
                f"{dimension['label']} "
                f"[{dimension['role']}]"
            )

            for index, status in enumerate(
                allowed
            ):

                print(
                    f"  {index} = {status}"
                )

            while True:

                choice = input("> ").strip()

                try:

                    choice = int(choice)

                    if (
                        0
                        <= choice
                        < len(allowed)
                    ):

                        break

                except ValueError:
                    pass

                print(
                    "Choix invalide."
                )

            annotation[
                dimension["id"]
            ] = allowed[choice]

        file_exists = (
            corpus_path.exists()
        )

        with corpus_path.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=columns
            )

            if not file_exists:

                writer.writeheader()

            writer.writerow(
                annotation
            )

        print(
            "\nAnnotation sauvegardée."
        )

    print()
    print(
        "Toutes les annotations "
        "disponibles sont terminées."
    )


if __name__ == "__main__":
    main()
