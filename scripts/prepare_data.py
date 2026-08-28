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

from dimension_predictor.data import (
    split_dataframe
)


def main():

    config = load_config()
    labels = load_labels()

    df = load_corpus(
        config,
        labels
    )

    validation = (
        validate_dataframe(
            df,
            config,
            labels
        )
    )

    if not validation["valid"]:

        print(
            "Le corpus contient "
            "des erreurs."
        )

        print(
            "Lancez : "
            "python scripts/"
            "validate_corpus.py"
        )

        raise SystemExit(1)

    train_df, val_df, test_df = (
        split_dataframe(
            df,
            config,
            labels
        )
    )

    output = (
        ROOT
        / "data"
        / "processed"
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        output / "train.csv",
        index=False
    )

    val_df.to_csv(
        output / "validation.csv",
        index=False
    )

    test_df.to_csv(
        output / "test.csv",
        index=False
    )

    print()
    print("PRÉPARATION TERMINÉE")
    print("=" * 60)

    print(
        "Train      :",
        len(train_df)
    )

    print(
        "Validation :",
        len(val_df)
    )

    print(
        "Test       :",
        len(test_df)
    )


if __name__ == "__main__":
    main()
