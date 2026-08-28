import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from dimension_predictor.labels import (
    get_allowed_statuses
)


def test_must_rapport():

    labels = {
        "status_must": [
            "validee",
            "manquante",
            "preuve_manquante",
            "absent"
        ],
        "status_should": [
            "favorable",
            "en_effort",
            "defavorable",
            "absent"
        ],
        "status_must_by_source_role": {
            "rapport": [
                "manquante",
                "preuve_manquante",
                "absent"
            ]
        }
    }

    dimension = {
        "role": "MUST"
    }

    statuses = get_allowed_statuses(
        labels,
        dimension,
        "rapport"
    )

    assert "validee" not in statuses
    assert "manquante" in statuses
