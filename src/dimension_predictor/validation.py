from collections import Counter

from .labels import (
    get_allowed_statuses
)


def validate_dataframe(
    df,
    config,
    labels_config
):

    errors = []
    warnings = []

    text_column = (
        config["data"]["text_column"]
    )

    source_column = (
        config["data"]["source_column"]
    )

    allowed_sources = set(
        config["data"]["allowed_sources"]
    )

    # --------------------------------------------------------
    # Textes vides
    # --------------------------------------------------------

    empty_mask = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    for index in df.index[empty_mask]:

        errors.append(
            f"Ligne {index + 2} : texte vide."
        )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    for index, source in df[source_column].items():

        if source not in allowed_sources:

            errors.append(
                f"Ligne {index + 2} : "
                f"source invalide '{source}'."
            )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    for index, row in df.iterrows():

        source = row[source_column]

        for dimension in labels_config["dimensions"]:

            dimension_id = dimension["id"]

            status = row[dimension_id]

            if status is None:
                status = ""

            status = str(status).strip()

            allowed = get_allowed_statuses(
                labels_config,
                dimension,
                source
            )

            if status not in allowed:

                errors.append(
                    f"Ligne {index + 2} / "
                    f"{dimension_id} : "
                    f"'{status}' interdit. "
                    f"Autorisés : {allowed}"
                )

    # --------------------------------------------------------
    # Doublons
    # --------------------------------------------------------

    duplicated = df.duplicated(
        subset=[text_column],
        keep=False
    )

    duplicate_count = int(
        duplicated.sum()
    )

    if duplicate_count:

        warnings.append(
            f"{duplicate_count} lignes "
            "appartiennent à des textes dupliqués."
        )

    # --------------------------------------------------------
    # Classes rares
    # --------------------------------------------------------

    for dimension in labels_config["dimensions"]:

        dimension_id = dimension["id"]

        counts = Counter(
            df[dimension_id]
            .fillna("")
            .astype(str)
        )

        for status, count in counts.items():

            if status != "absent" and count < 10:

                warnings.append(
                    f"{dimension_id} / {status} : "
                    f"seulement {count} exemples."
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
