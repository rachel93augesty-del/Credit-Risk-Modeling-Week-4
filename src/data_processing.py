import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import scorecardpy as sc  # For WoE encoding


# ---------------------------
# Utility Functions
# ---------------------------

def get_numeric_columns(df: pd.DataFrame):
    """Return list of numeric column names."""
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame):
    """Return list of categorical column names."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0):
    """Return a boolean mask where True indicates an outlier based on z-score."""
    mean_val = df[column].mean()
    std_val = df[column].std()
    return ((df[column] - mean_val).abs() > threshold * std_val)


# ---------------------------
# Custom Transformers
# ---------------------------

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Creates customer-level aggregate features from transaction-level data."""

    def __init__(self, customer_col="CustomerId", amount_col="Amount"):
        self.customer_col = customer_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = (
            X.groupby(self.customer_col)[self.amount_col]
            .agg(
                [
                    ("total_amount", "sum"),
                    ("avg_amount", "mean"),
                    ("transaction_count", "count"),
                    ("std_amount", "std"),
                ]
            )
            .reset_index()
        )
        return agg_df


class DateFeatures(BaseEstimator, TransformerMixin):
    """Extracts year, month, day, and hour from a datetime column."""

    def __init__(self, date_col="TransactionStartTime"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.date_col] = pd.to_datetime(X_[self.date_col])
        X_["transaction_year"] = X_[self.date_col].dt.year
        X_["transaction_month"] = X_[self.date_col].dt.month
        X_["transaction_day"] = X_[self.date_col].dt.day
        X_["transaction_hour"] = X_[self.date_col].dt.hour
        return X_.drop(columns=[self.date_col])


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical variables using WoE transformation."""

    def __init__(self, target="FraudResult"):
        self.target = target
        self.woe_bins = None

    def fit(self, X, y=None):
        df = X.copy()
        df[self.target] = y
        cat_cols = get_categorical_columns(df)
        self.woe_bins = sc.woebin(df, y=self.target, x=cat_cols)
        return self

    def transform(self, X):
        return sc.woebin_ply(X, self.woe_bins)


# ---------------------------
# Preprocessing Pipeline
# ---------------------------

def build_preprocessing_pipeline(
    numeric_features, categorical_features, date_feature="TransactionStartTime"
):
    """Builds a full preprocessing pipeline for feature engineering."""

    # Numeric pipeline: impute + scale
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Categorical pipeline: impute + One-Hot
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine numeric & categorical pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # Full pipeline including date features
    full_pipeline = Pipeline(
        steps=[("date_features", DateFeatures(date_col=date_feature)), ("preprocessor", preprocessor)]
    )

    return full_pipeline


# ---------------------------
# Example Usage
# ---------------------------

if __name__ == "__main__":
    df = pd.read_csv("../data/transactions.csv")

    numeric_features = ["Amount", "Value"]
    categorical_features = ["CurrencyCode", "CountryCode", "ProviderId", "ProductCategory", "ChannelId"]

    pipeline = build_preprocessing_pipeline(numeric_features, categorical_features)
    df_transformed = pipeline.fit_transform(df)

    print(df_transformed.shape)
