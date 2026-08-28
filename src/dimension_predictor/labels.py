def get_dimensions(labels_config):

    return labels_config["dimensions"]


def get_dimension_ids(labels_config):

    return [
        dimension["id"]
        for dimension in labels_config["dimensions"]
    ]


def get_dimension_labels(labels_config):

    return [
        dimension["label"]
        for dimension in labels_config["dimensions"]
    ]


def get_dimension_by_id(labels_config, dimension_id):

    for dimension in labels_config["dimensions"]:

        if dimension["id"] == dimension_id:
            return dimension

    raise KeyError(
        f"Dimension inconnue : {dimension_id}"
    )


def get_allowed_statuses(
    labels_config,
    dimension,
    source=None
):

    role = dimension["role"]

    if role == "MUST":

        if source is not None:

            rules = labels_config.get(
                "status_must_by_source_role",
                {}
            )

            if source in rules:
                return rules[source]

        return labels_config["status_must"]

    if role == "SHOULD":
        return labels_config["status_should"]

    raise ValueError(
        f"Rôle inconnu : {role}"
    )


def get_non_absent_statuses(
    labels_config,
    dimension,
    source=None
):

    return [
        status
        for status in get_allowed_statuses(
            labels_config,
            dimension,
            source
        )
        if status != "absent"
    ]
