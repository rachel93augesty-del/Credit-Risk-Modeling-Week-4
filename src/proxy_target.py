# src/proxy_target.py

import pandas as pd
import numpy as np

# ---------------------------
# RFM Calculation
# ---------------------------


def calculate_rfm(
    df, customer_col="CustomerId", date_col="TransactionStartTime", amount_col="Amount", snapshot_date=None
):
    """
    Calculate RFM metrics for each customer:
    - Recency: Days since last transaction
    - Frequency: Number of transactions
    - Monetary: Total transaction amount
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Set snapshot date as max date + 1 day if not provided
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    # Group by customer
    rfm = (
        df.groupby(customer_col)
        .agg(
            recency_days=(date_col, lambda x: (snapshot_date - x.max()).days),
            frequency=(date_col, "count"),
            monetary=(amount_col, "sum"),
        )
        .reset_index()
    )

    return rfm


# ---------------------------
# High-Risk Customer Labeling
# ---------------------------


def label_high_risk_customers(rfm_df, recency_thresh=90, frequency_thresh=2, monetary_thresh=0):
    """
    Label customers as high-risk (1) or low-risk (0) based on RFM thresholds.
    - recency_thresh: maximum days since last transaction to be low-risk
    - frequency_thresh: minimum number of transactions to be low-risk
    - monetary_thresh: minimum total transaction amount to be low-risk
    """
    df = rfm_df.copy()
    df["high_risk"] = np.where(
        (df["recency_days"] > recency_thresh)
        | (df["frequency"] < frequency_thresh)
        | (df["monetary"] <= monetary_thresh),
        1,
        0,
    )
    return df


# ---------------------------
# Save RFM Data
# ---------------------------


def save_rfm(rfm_df, path="../data/processed/rfm_features.csv"):
    """Save the RFM dataframe to a CSV file."""
    rfm_df.to_csv(path, index=False)
    print(f"RFM data saved to {path}")


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    df = pd.read_csv("../data/raw/raw_data.csv")

    rfm_df = calculate_rfm(df)
    rfm_df = label_high_risk_customers(rfm_df)
    save_rfm(rfm_df)
    print(rfm_df.head())
