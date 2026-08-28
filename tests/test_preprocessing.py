import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from dimension_predictor.preprocessing import (
    clean_text
)


def test_clean_text():

    assert clean_text(
        " Bonjour    monde "
    ) == "Bonjour monde"
