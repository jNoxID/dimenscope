import json
import sys
import time

from pathlib import Path

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from dimension_predictor.config import (
    load_config,
    load_labels
)

from dimension_predictor.dataset import (
    MultiDimensionDataset
)

from dimension_predictor.collator import (
    MultiDimensionCollator
)

from dimension_predictor.model import (
    MultiDimensionTransformer
)

from dimension_predictor.evaluation import (
    evaluate_model
)

from dimension_predictor.utils import (
    seed_everything,
    get_device,
    save_json
)


STATUS_VOCABULARY = [
    "validee",
    "manquante",
    "preuve_manquante",
    "favorable",
    "en_effort",
    "defavorable",
]


def main():

    config = load_config()
    labels = load_labels()

    seed_everything(
        config["seed"]
    )

    processed = (
        ROOT
        / "data"
        / "processed"
    )

    train_path = (
        processed / "train.csv"
    )

    val_path = (
        processed
        / "validation.csv"
    )

    test_path = (
        processed / "test.csv"
    )

    for path in [
        train_path,
        val_path,
        test_path
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "Données préparées "
                "introuvables. Lancez "
                "d'abord : "
                "python scripts/"
                "prepare_data.py"
            )

    train_df = pd.read_csv(
        train_path
    )

    val_df = pd.read_csv(
        val_path
    )

    test_df = pd.read_csv(
        test_path
    )

    model_key = (
        config["model"]["selected"]
    )

    model_name = (
        config["model"][model_key]
    )

    method = (
        config["model"]["method"]
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_name,
            use_fast=True
        )
    )

    train_dataset = (
        MultiDimensionDataset(
            train_df,
            tokenizer,
            config,
            labels
        )
    )

    val_dataset = (
        MultiDimensionDataset(
            val_df,
            tokenizer,
            config,
            labels
        )
    )

    test_dataset = (
        MultiDimensionDataset(
            test_df,
            tokenizer,
            config,
            labels
        )
    )

    model = (
        MultiDimensionTransformer(
            model_name=model_name,
            num_dimensions=len(
                labels["dimensions"]
            ),
            num_status_classes=len(
                STATUS_VOCABULARY
            ),
            method=method,
            dropout=(
                config["model"][
                    "dropout"
                ]
            ),
            lora_config=(
                config.get("lora")
            ),
            presence_loss_weight=(
                config["training"][
                    "presence_loss_weight"
                ]
            ),
            status_loss_weight=(
                config["training"][
                    "status_loss_weight"
                ]
            )
        )
    )

    if method == "full":

        lr = config["training"][
            "learning_rate_full"
        ]

    elif method == "lora":

        lr = config["training"][
            "learning_rate_lora"
        ]

    else:

        lr = config["training"][
            "learning_rate_frozen"
        ]

    checkpoint_dir = (
        ROOT
        / config["artifacts"][
            "checkpoints"
        ]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    use_fp16 = (
        torch.cuda.is_available()
        and not use_bf16
    )

    args = TrainingArguments(

        output_dir=str(
            checkpoint_dir
        ),

        num_train_epochs=(
            config["training"][
                "epochs"
            ]
        ),

        per_device_train_batch_size=(
            config["training"][
                "batch_size"
            ]
        ),

        per_device_eval_batch_size=(
            config["training"][
                "eval_batch_size"
            ]
        ),

        learning_rate=lr,

        weight_decay=(
            config["training"][
                "weight_decay"
            ]
        ),

        warmup_ratio=(
            config["training"][
                "warmup_ratio"
            ]
        ),

        lr_scheduler_type="cosine",

        max_grad_norm=1.0,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model=(
            "eval_loss"
        ),

        greater_is_better=False,

        save_total_limit=2,

        fp16=use_fp16,
        bf16=use_bf16,

        report_to="none",

        seed=config["seed"],
        data_seed=config["seed"]
    )

    collator = (
        MultiDimensionCollator(
            tokenizer
        )
    )

    trainer = Trainer(

        model=model,

        args=args,

        train_dataset=(
            train_dataset
        ),

        eval_dataset=(
            val_dataset
        ),

        data_collator=(
            collator
        ),

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=(
                    config[
                        "training"
                    ][
                        "early_stopping_patience"
                    ]
                )
            )
        ]
    )

    device = get_device()

    print()
    print("=" * 70)
    print("ENTRAÎNEMENT")
    print("=" * 70)

    print(
        "Modèle :",
        model_name
    )

    print(
        "Méthode :",
        method
    )

    print(
        "Device :",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU :",
            torch.cuda.get_device_name(
                0
            )
        )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        f"Paramètres entraînables : "
        f"{trainable:,} / "
        f"{total:,}"
    )

    start = time.perf_counter()

    trainer.train()

    duration = (
        time.perf_counter()
        - start
    )

    model.to(device)

    print(
        "\nÉvaluation validation..."
    )

    val_metrics = evaluate_model(
        model,
        val_dataset,
        collator,
        device,
        config["training"][
            "eval_batch_size"
        ]
    )

    print(
        json.dumps(
            val_metrics,
            indent=2,
            ensure_ascii=False
        )
    )

    print(
        "\nÉvaluation test..."
    )

    test_metrics = evaluate_model(
        model,
        test_dataset,
        collator,
        device,
        config["training"][
            "eval_batch_size"
        ]
    )

    print(
        json.dumps(
            test_metrics,
            indent=2,
            ensure_ascii=False
        )
    )

    final_dir = (
        ROOT
        / config["artifacts"][
            "final_model"
        ]
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Sauvegarde du state_dict.
    torch.save(
        model.state_dict(),
        final_dir / "model.pt"
    )

    tokenizer.save_pretrained(
        final_dir / "tokenizer"
    )

    metadata = {
        "model_name": model_name,
        "model_key": model_key,
        "method": method,
        "num_dimensions": len(
            labels["dimensions"]
        ),
        "status_vocabulary": (
            STATUS_VOCABULARY
        ),
        "train_time_seconds": (
            round(duration, 2)
        ),
        "trainable_parameters": (
            trainable
        ),
        "total_parameters": (
            total
        ),
        "validation_metrics": (
            val_metrics
        ),
        "test_metrics": (
            test_metrics
        )
    }

    save_json(
        metadata,
        final_dir
        / "metadata.json"
    )

    print()
    print(
        "Modèle sauvegardé dans :"
    )

    print(final_dir)


if __name__ == "__main__":
    main()
