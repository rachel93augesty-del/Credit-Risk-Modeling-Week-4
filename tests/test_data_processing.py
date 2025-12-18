import sys
from pathlib import Path

import pandas as pd

# Add src/ to Python path (required for pytest)
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from src.data_processing import (  # noqa: E402
    get_numeric_columns,
    get_categorical_columns,
)


def test_get_numeric_columns():
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3],
            "num2": [4.0, 5.0, 6.0],
            "cat": ["a", "b", "c"],
        }
    )

    numeric_cols = get_numeric_columns(df)

    assert numeric_cols == ["num1", "num2"]


def test_get_categorical_columns():
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3],
            "num2": [4.0, 5.0, 6.0],
            "cat": ["a", "b", "c"],
        }
    )

    categorical_cols = get_categorical_columns(df)

    assert categorical_cols == ["cat"]
