import torch


STATUS_VOCABULARY = [
    "validee",
    "manquante",
    "preuve_manquante",
    "favorable",
    "en_effort",
    "defavorable",
]


@torch.inference_mode()
def predict(
    text,
    source,
    model,
    tokenizer,
    labels_config,
    device,
    max_length=128
):

    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    encoded = {
        key: value.to(device)
        for key, value
        in encoded.items()
    }

    outputs = model(**encoded)

    (
        presence_logits,
        status_logits
    ) = outputs["logits"]

    presence_prob = torch.softmax(
        presence_logits,
        dim=-1
    )[0]

    status_prob = torch.softmax(
        status_logits,
        dim=-1
    )[0]

    results = []

    for index, dimension in enumerate(
        labels_config["dimensions"]
    ):

        present_probability = float(
            presence_prob[
                index,
                1
            ].cpu()
        )

        absent_probability = float(
            presence_prob[
                index,
                0
            ].cpu()
        )

        is_present = (
            present_probability
            >= absent_probability
        )

        if not is_present:

            results.append({
                "id": dimension["id"],
                "label": dimension["label"],
                "role": dimension["role"],
                "presence": "absent",
                "presence_confidence": (
                    absent_probability
                ),
                "status": "absent",
                "status_confidence": (
                    absent_probability
                )
            })

            continue

        # Statuts autorisés
        if dimension["role"] == "MUST":

            allowed = (
                labels_config
                .get(
                    "status_must_by_source_role",
                    {}
                )
                .get(
                    source,
                    labels_config[
                        "status_must"
                    ]
                )
            )

        else:

            allowed = labels_config[
                "status_should"
            ]

        allowed = [
            status
            for status in allowed
            if status != "absent"
        ]

        candidates = []

        for status in allowed:

            if status not in (
                STATUS_VOCABULARY
            ):
                continue

            status_index = (
                STATUS_VOCABULARY.index(
                    status
                )
            )

            probability = float(
                status_prob[
                    index,
                    status_index
                ].cpu()
            )

            candidates.append(
                (
                    status,
                    probability
                )
            )

        if candidates:

            predicted_status, confidence = (
                max(
                    candidates,
                    key=lambda x: x[1]
                )
            )

        else:

            predicted_status = "absent"
            confidence = 0.0

        results.append({
            "id": dimension["id"],
            "label": dimension["label"],
            "role": dimension["role"],
            "presence": "present",
            "presence_confidence": (
                present_probability
            ),
            "status": predicted_status,
            "status_confidence": (
                confidence
            )
        })

    return results
