import numpy as np
import torch

from torch.utils.data import (
    DataLoader
)

from .metrics import (
    compute_metrics_arrays
)


@torch.inference_mode()
def evaluate_model(
    model,
    dataset,
    collator,
    device,
    batch_size=32
):

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=(
            torch.cuda.is_available()
        )
    )

    presence_true = []
    presence_pred = []

    status_true = []
    status_pred = []

    for batch in loader:

        batch = {
            key: value.to(
                device,
                non_blocking=True
            )
            for key, value
            in batch.items()
        }

        outputs = model(**batch)

        (
            presence_logits,
            status_logits
        ) = outputs["logits"]

        presence_true.append(
            batch[
                "presence_labels"
            ]
            .cpu()
            .numpy()
        )

        status_true.append(
            batch[
                "status_labels"
            ]
            .cpu()
            .numpy()
        )

        presence_pred.append(
            presence_logits
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )

        status_pred.append(
            status_logits
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )

    return compute_metrics_arrays(
        np.concatenate(
            presence_true
        ),
        np.concatenate(
            presence_pred
        ),
        np.concatenate(
            status_true
        ),
        np.concatenate(
            status_pred
        )
    )
