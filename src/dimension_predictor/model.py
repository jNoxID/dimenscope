import torch
import torch.nn as nn

from transformers import AutoModel

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)


IGNORE_INDEX = -100

NUM_PRESENCE_CLASSES = 2


class MultiDimensionTransformer(
    nn.Module
):

    def __init__(
        self,
        model_name,
        num_dimensions,
        num_status_classes,
        method="full",
        dropout=0.15,
        lora_config=None,
        presence_loss_weight=1.0,
        status_loss_weight=1.0,
    ):

        super().__init__()

        self.num_dimensions = (
            num_dimensions
        )

        self.num_status_classes = (
            num_status_classes
        )

        self.presence_loss_weight = (
            presence_loss_weight
        )

        self.status_loss_weight = (
            status_loss_weight
        )

        self.encoder = (
            AutoModel.from_pretrained(
                model_name
            )
        )

        # ----------------------------------------------------
        # Transfer learning
        # ----------------------------------------------------

        if method == "frozen":

            for parameter in (
                self.encoder.parameters()
            ):

                parameter.requires_grad = False

        elif method == "lora":

            cfg = lora_config or {}

            peft_cfg = LoraConfig(
                task_type=(
                    TaskType.FEATURE_EXTRACTION
                ),
                r=cfg.get("r", 8),
                lora_alpha=cfg.get(
                    "alpha",
                    16
                ),
                lora_dropout=cfg.get(
                    "dropout",
                    0.05
                ),
                target_modules=cfg.get(
                    "target_modules",
                    [
                        "query",
                        "value"
                    ]
                ),
                bias="none"
            )

            self.encoder = (
                get_peft_model(
                    self.encoder,
                    peft_cfg
                )
            )

        elif method != "full":

            raise ValueError(
                f"Méthode inconnue : "
                f"{method}"
            )

        hidden = (
            self.encoder.config.hidden_size
        )

        self.normalization = (
            nn.LayerNorm(hidden)
        )

        self.dropout = (
            nn.Dropout(dropout)
        )

        # ----------------------------------------------------
        # Présence
        # ----------------------------------------------------

        self.presence_classifier = (
            nn.Linear(
                hidden,
                num_dimensions
                * NUM_PRESENCE_CLASSES
            )
        )

        # ----------------------------------------------------
        # Statut
        # ----------------------------------------------------

        # La branche statut reçoit :
        # représentation texte +
        # probabilité de présence
        status_input_size = (
            hidden + num_dimensions
        )

        self.status_hidden = (
            nn.Sequential(
                nn.Linear(
                    status_input_size,
                    hidden
                ),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        )

        self.status_classifier = (
            nn.Linear(
                hidden,
                num_dimensions
                * num_status_classes
            )
        )

    @staticmethod
    def mean_pooling(
        hidden_state,
        attention_mask
    ):

        mask = (
            attention_mask
            .unsqueeze(-1)
            .type_as(hidden_state)
        )

        summed = (
            hidden_state * mask
        ).sum(dim=1)

        count = (
            mask
            .sum(dim=1)
            .clamp(min=1e-9)
        )

        return summed / count

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        presence_labels=None,
        status_labels=None,
        **kwargs
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        pooled = self.normalization(
            pooled
        )

        pooled = self.dropout(
            pooled
        )

        # ----------------------------------------------------
        # Présence
        # ----------------------------------------------------

        presence_logits = (
            self.presence_classifier(
                pooled
            )
            .view(
                -1,
                self.num_dimensions,
                NUM_PRESENCE_CLASSES
            )
        )

        presence_prob = (
            torch.softmax(
                presence_logits,
                dim=-1
            )[:, :, 1]
        )

        # ----------------------------------------------------
        # Statut conditionné par présence
        # ----------------------------------------------------

        status_input = torch.cat(
            [
                pooled,
                presence_prob
            ],
            dim=-1
        )

        status_hidden = (
            self.status_hidden(
                status_input
            )
        )

        status_logits = (
            self.status_classifier(
                status_hidden
            )
            .view(
                -1,
                self.num_dimensions,
                self.num_status_classes
            )
        )

        loss = None

        presence_loss = None
        status_loss = None

        if presence_labels is not None:

            presence_loss = (
                nn.functional.cross_entropy(
                    presence_logits.reshape(
                        -1,
                        NUM_PRESENCE_CLASSES
                    ),
                    presence_labels.reshape(
                        -1
                    )
                )
            )

        if status_labels is not None:

            status_loss = (
                nn.functional.cross_entropy(
                    status_logits.reshape(
                        -1,
                        self.num_status_classes
                    ),
                    status_labels.reshape(
                        -1
                    ),
                    ignore_index=IGNORE_INDEX
                )
            )

        if (
            presence_loss is not None
            and status_loss is not None
        ):

            loss = (
                self.presence_loss_weight
                * presence_loss
                +
                self.status_loss_weight
                * status_loss
            )

        return {
            "loss": loss,
            "presence_loss": presence_loss,
            "status_loss": status_loss,
            "logits": (
                presence_logits,
                status_logits
            )
        }
