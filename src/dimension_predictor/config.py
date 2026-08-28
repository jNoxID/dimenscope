from pathlib import Path
import json
import yaml


def load_config(path="configs/config.yaml"):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(path="configs/labels.json"):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Labels introuvables : {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
