import numpy as np

from sklearn.metrics import (
    f1_score
)


IGNORE_INDEX = -100


def compute_metrics_arrays(
    presence_true,
    presence_pred,
    status_true,
    status_pred
):

    num_dimensions = (
        presence_true.shape[1]
    )

    presence_scores = []
    status_scores = []

    per_dimension = []

    for index in range(
        num_dimensions
    ):

        p_f1 = f1_score(
            presence_true[:, index],
            presence_pred[:, index],
            average="macro",
            zero_division=0
        )

        mask = (
            status_true[:, index]
            != IGNORE_INDEX
        )

        if mask.any():

            s_f1 = f1_score(
                status_true[
                    mask,
                    index
                ],
                status_pred[
                    mask,
                    index
                ],
                average="macro",
                zero_division=0
            )

            status_scores.append(
                s_f1
            )

        else:

            s_f1 = None

        presence_scores.append(
            p_f1
        )

        per_dimension.append({
            "index": index,
            "f1_presence": float(
                p_f1
            ),
            "f1_status": (
                float(s_f1)
                if s_f1 is not None
                else None
            ),
            "n_present": int(
                mask.sum()
            )
        })

    f1_presence = float(
        np.mean(
            presence_scores
        )
    )

    f1_status = float(
        np.mean(
            status_scores
        )
    ) if status_scores else 0.0

    return {
        "f1_presence": f1_presence,
        "f1_status": f1_status,
        "combined_f1": float(
            (
                f1_presence
                + f1_status
            ) / 2
        ),
        "per_dimension": (
            per_dimension
        )
    }
