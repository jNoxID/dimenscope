from dataclasses import dataclass

import torch

from transformers import (
    DataCollatorWithPadding
)


@dataclass
class MultiDimensionCollator:

    tokenizer: object

    def __post_init__(self):

        self.padding_collator = (
            DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=(
                    8
                    if torch.cuda.is_available()
                    else None
                )
            )
        )

    def __call__(self, features):

        presence = [
            feature["presence_labels"]
            for feature in features
        ]

        status = [
            feature["status_labels"]
            for feature in features
        ]

        clean = []

        for feature in features:

            clean.append({
                key: value
                for key, value
                in feature.items()
                if key not in {
                    "presence_labels",
                    "status_labels"
                }
            })

        batch = self.padding_collator(
            clean
        )

        batch["presence_labels"] = (
            torch.tensor(
                presence,
                dtype=torch.long
            )
        )

        batch["status_labels"] = (
            torch.tensor(
                status,
                dtype=torch.long
            )
        )

        return batch
