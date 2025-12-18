# tests/test_data_processing.py

import sys
import os
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import (
    get_numeric_columns,
    get_categorical_columns,
    detect_outliers
)

# -------------------------------
# Sample DataFrame for testing
# -------------------------------
def sample_df():
    data = {
        'Recency': [10, 20, 30, 40, 50, 60],
        'Frequency': [1, 2, 3, 4, 5, 6],
        'Monetary': [100, 200, 300, 400, 500, 600],
        'Cluster': [0, 1, 0, 1, 0, 1],
        'Category': ['A', 'B', 'A', 'B', 'C', 'C']
    }
    return pd.DataFrame(data)


# -------------------------------
# Test 1: get_numeric_columns
# -------------------------------
def test_get_numeric_columns():
    df = sample_df()
    numeric_cols = get_numeric_columns(df)
    expected = ['Recency', 'Frequency', 'Monetary', 'Cluster']
    assert set(numeric_cols) == set(expected), f"Expected {expected}, got {numeric_cols}"


# -------------------------------
# Test 2: get_categorical_columns
# -------------------------------
def test_get_categorical_columns():
    df = sample_df()
    cat_cols = get_categorical_columns(df)
    expected = ['Category']
    assert cat_cols == expected, f"Expected {expected}, got {cat_cols}"


# -------------------------------
# Test 3: detect_outliers
# -------------------------------
def test_detect_outliers(capsys):
    df = sample_df()
    # Introduce an extreme outlier
    df.loc[0, 'Monetary'] = 10000
    detect_outliers(df)
    captured = capsys.readouterr()
    assert "Monetary: 1 outliers" in captured.out, "Outlier detection failed for Monetary column"
