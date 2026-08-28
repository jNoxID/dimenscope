from pathlib import Path

import pandas as pd

from .preprocessing import clean_text
from .labels import get_dimension_ids


def load_corpus(config, labels_config):

    path = Path(
        config["data"]["corpus"]
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Corpus introuvable : {path}"
        )

    df = pd.read_csv(path)

    required = [
        "id",
        config["data"]["text_column"],
        config["data"]["source_column"],
    ] + get_dimension_ids(labels_config)

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(missing)
        )

    text_column = (
        config["data"]["text_column"]
    )

    df[text_column] = (
        df[text_column]
        .fillna("")
        .apply(clean_text)
    )

    return df
