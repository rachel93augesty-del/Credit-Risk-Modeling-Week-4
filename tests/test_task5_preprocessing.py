# tests/test_task5_preprocessing.py

import sys
import os
import pandas as pd
import pytest

# Add project root to path so src modules are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import DateFeatures, build_preprocessing_pipeline

# -------------------------------
# Sample DataFrame for testing
# -------------------------------
def sample_df():
    data = {
        'TransactionStartTime': [
            '2025-01-01 08:00:00',
            '2025-02-15 14:30:00',
            '2025-03-20 09:15:00'
        ],
        'Amount': [100, 200, 150],
        'Value': [10, 20, 15],
        'CurrencyCode': ['USD', 'EUR', 'USD'],
        'CountryCode': ['US', 'DE', 'US'],
        'ProviderId': ['P1', 'P2', 'P1'],
        'ProductCategory': ['A', 'B', 'A'],
        'ChannelId': ['C1', 'C2', 'C1']
    }
    return pd.DataFrame(data)

# -------------------------------
# Test 1: DateFeatures transformer
# -------------------------------
def test_date_features():
    df = sample_df()
    transformer = DateFeatures(date_col='TransactionStartTime')
    transformed = transformer.fit_transform(df)
    
    # Check new columns exist
    for col in ['transaction_year', 'transaction_month', 'transaction_day', 'transaction_hour']:
        assert col in transformed.columns, f"{col} should be in transformed DataFrame"
    
    # Original column should be dropped
    assert 'TransactionStartTime' not in transformed.columns

# -------------------------------
# Test 2: Preprocessing pipeline
# -------------------------------
def test_preprocessing_pipeline():
    df = sample_df()
    numeric_features = ['Amount', 'Value']
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory', 'ChannelId']
    
    pipeline = build_preprocessing_pipeline(numeric_features, categorical_features)
    transformed = pipeline.fit_transform(df)
    
    # Should return a numpy array
    import numpy as np
    assert isinstance(transformed, np.ndarray), "Pipeline output should be a numpy array"
    
    # Number of columns should match numeric + one-hot encoded categorical
    expected_num_cols = len(numeric_features) + sum(df[col].nunique() for col in categorical_features)
    assert transformed.shape[1] == expected_num_cols, f"Expected {expected_num_cols} columns after transformation"

# -------------------------------
# Run tests via pytest
# -------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
