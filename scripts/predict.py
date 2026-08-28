import argparse
import json
import sys

from pathlib import Path

import torch

from transformers import (
    AutoTokenizer
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

from dimension_predictor.model import (
    MultiDimensionTransformer
)

from dimension_predictor.inference import (
    predict
)

from dimension_predictor.utils import (
    get_device
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        required=True
    )

    parser.add_argument(
        "--source",
        choices=[
            "rapport",
            "entreprise"
        ],
        required=True
    )

    args = parser.parse_args()

    config = load_config()
    labels = load_labels()

    final_dir = (
        ROOT
        / config["artifacts"][
            "final_model"
        ]
    )

    metadata_path = (
        final_dir
        / "metadata.json"
    )

    model_path = (
        final_dir
        / "model.pt"
    )

    if (
        not metadata_path.exists()
        or not model_path.exists()
    ):

        raise FileNotFoundError(
            "Aucun modèle entraîné. "
            "Lancez d'abord "
            "python scripts/train.py"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            final_dir
            / "tokenizer"
        )
    )

    model = (
        MultiDimensionTransformer(
            model_name=(
                metadata[
                    "model_name"
                ]
            ),
            num_dimensions=(
                metadata[
                    "num_dimensions"
                ]
            ),
            num_status_classes=len(
                metadata[
                    "status_vocabulary"
                ]
            ),
            method=(
                metadata["method"]
            ),
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

    state = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(
        state
    )

    device = get_device()

    model.to(device)

    results = predict(
        text=args.text,
        source=args.source,
        model=model,
        tokenizer=tokenizer,
        labels_config=labels,
        device=device,
        max_length=(
            config["model"][
                "max_length"
            ]
        )
    )

    print()
    print("=" * 80)
    print("ANALYSE")
    print("=" * 80)

    print()
    print("Texte :")
    print(args.text)

    print()
    print(
        "Source :",
        args.source
    )

    print()
    print("-" * 80)

    for result in results:

        print()
        print(
            result["label"],
            f"[{result['role']}]"
        )

        print(
            "  Présence :",
            result["presence"],
            f"({result['presence_confidence']:.1%})"
        )

        print(
            "  Statut   :",
            result["status"],
            f"({result['status_confidence']:.1%})"
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
