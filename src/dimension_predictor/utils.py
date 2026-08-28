import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def ensure_directory(path):

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def save_json(data, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def load_json(path):

    with Path(path).open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)
