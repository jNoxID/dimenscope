import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def create_stratification_key(
    df,
    labels_config,
    source_column
):

    dimension_ids = [
        dimension["id"]
        for dimension in labels_config["dimensions"]
    ]

    presence_count = (
        df[dimension_ids] != "absent"
    ).sum(axis=1)

    bucket = pd.cut(
        presence_count,
        bins=[
            -1,
            0,
            2,
            5,
            np.inf
        ],
        labels=[
            "0",
            "1-2",
            "3-5",
            "6+"
        ]
    ).astype(str)

    return (
        df[source_column].astype(str)
        + "_"
        + bucket
    )


def safe_stratify(series):

    counts = series.value_counts()

    if len(counts) < 2:
        return None

    if counts.min() < 2:
        return None

    return series


def split_dataframe(
    df,
    config,
    labels_config
):

    seed = config["seed"]

    source_column = (
        config["data"]["source_column"]
    )

    validation_size = (
        config["data"]["validation_size"]
    )

    test_size = (
        config["data"]["test_size"]
    )

    temp_size = (
        validation_size
        + test_size
    )

    key = create_stratification_key(
        df,
        labels_config,
        source_column
    )

    stratify = safe_stratify(key)

    if stratify is None:
        stratify = safe_stratify(
            df[source_column]
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=seed,
        stratify=stratify
    )

    relative_test = (
        test_size / temp_size
    )

    temp_key = create_stratification_key(
        temp_df,
        labels_config,
        source_column
    )

    temp_stratify = safe_stratify(
        temp_key
    )

    if temp_stratify is None:

        temp_stratify = safe_stratify(
            temp_df[source_column]
        )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_stratify
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
