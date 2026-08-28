import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from dimension_predictor.config import (
    load_config,
    load_labels
)

from dimension_predictor.corpus import (
    load_corpus
)

from dimension_predictor.validation import (
    validate_dataframe
)


def main():

    config = load_config()
    labels = load_labels()

    df = load_corpus(
        config,
        labels
    )

    result = validate_dataframe(
        df,
        config,
        labels
    )

    print()
    print("=" * 70)
    print("VALIDATION DU CORPUS")
    print("=" * 70)

    print(
        f"\nNombre de textes : "
        f"{len(df)}"
    )

    print(
        f"Nombre de dimensions : "
        f"{len(labels['dimensions'])}"
    )

    print(
        f"\nErreurs : "
        f"{len(result['errors'])}"
    )

    for error in result["errors"]:

        print(
            "  [ERREUR]",
            error
        )

    print(
        f"\nAvertissements : "
        f"{len(result['warnings'])}"
    )

    for warning in result[
        "warnings"
    ]:

        print(
            "  [ATTENTION]",
            warning
        )

    print()

    if result["valid"]:

        print(
            "CORPUS VALIDE."
        )

    else:

        print(
            "CORPUS INVALIDE."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
