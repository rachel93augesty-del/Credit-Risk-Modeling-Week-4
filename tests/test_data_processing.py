import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest
from src.data_processing import get_numeric_columns, get_categorical_columns, detect_outliers


def test_get_numeric_columns():
    df = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [4.0, 5.0, 6.0],
        "cat": ["a", "b", "c"]
    })
    numeric_cols = get_numeric_columns(df)
    assert numeric_cols == ["num1", "num2"]

def test_get_categorical_columns():
    df = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [4.0, 5.0, 6.0],
        "cat": ["a", "b", "c"]
    })
    categorical_cols = get_categorical_columns(df)
    assert categorical_cols == ["cat"]
