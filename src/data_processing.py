 #task-3
# src/preprocessing.py

from pdb import main
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import scorecardpy as sc  # For WoE encoding


# ---------------------------
# Custom Transformers
# ---------------------------

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Creates customer-level aggregate features from transaction-level data."""
    
    def __init__(self, customer_col='CustomerId', amount_col='Amount'):
        self.customer_col = customer_col
        self.amount_col = amount_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        agg_df = X.groupby(self.customer_col)[self.amount_col].agg([
            ('total_amount', 'sum'),
            ('avg_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_amount', 'std')
        ]).reset_index()
        return agg_df


class DateFeatures(BaseEstimator, TransformerMixin):
    """Extracts year, month, day, and hour from a datetime column."""
    
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        X_[self.date_col] = pd.to_datetime(X_[self.date_col])
        X_['transaction_year'] = X_[self.date_col].dt.year
        X_['transaction_month'] = X_[self.date_col].dt.month
        X_['transaction_day'] = X_[self.date_col].dt.day
        X_['transaction_hour'] = X_[self.date_col].dt.hour
        return X_.drop(columns=[self.date_col])


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical variables using WoE transformation."""
    
    def __init__(self, target='FraudResult'):
        self.target = target
        self.woe_bins = None
        
    def fit(self, X, y=None):
        df = X.copy()
        df[self.target] = y
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        self.woe_bins = sc.woebin(df, y=self.target, x=cat_cols)
        return self
    
    def transform(self, X):
        return sc.woebin_ply(X, self.woe_bins)


# ---------------------------
# Main Preprocessing Pipeline
# ---------------------------

def build_preprocessing_pipeline(numeric_features, categorical_features, date_feature='TransactionStartTime'):
    """
    Builds a full preprocessing pipeline for Task 3 feature engineering.
    """

    # Numeric pipeline: impute + scale
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: impute + One-Hot
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine numeric & categorical pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Full pipeline including date features
    full_pipeline = Pipeline(steps=[
        ('date_features', DateFeatures(date_col=date_feature)),
        ('preprocessor', preprocessor)
        # You can append WoEEncoder here if target is available
        # ('woe', WoEEncoder(target='FraudResult'))
    ])
    
    return full_pipeline


# ---------------------------
# Example Usage
# ---------------------------

if __name__ == '__main__':
    df = pd.read_csv('../data/transactions.csv')
    
    # Define columns
    numeric_features = ['Amount', 'Value']
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory', 'ChannelId']
    
    pipeline = build_preprocessing_pipeline(numeric_features, categorical_features)
    df_transformed = pipeline.fit_transform(df)
    
    print(df_transformed.shape)

# src/data_processing.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set(style="whitegrid")  # Apply seaborn style globally

# -----------------------------
# Data Loading & Overview
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load CSV data and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def data_overview(df: pd.DataFrame):
    """Print basic info: rows, columns, dtypes, first 5 rows."""
    print("===== DATA OVERVIEW =====")
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
    print("\nColumn data types:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())

# -----------------------------
# Summary Statistics
# -----------------------------
def summarize_numeric(df: pd.DataFrame):
    """Display descriptive statistics for numeric columns."""
    print("===== NUMERIC SUMMARY =====")
    display(df.describe())  # noqa: F821

def summarize_categorical(df: pd.DataFrame):
    """Display counts of unique values for categorical columns."""
    print("===== CATEGORICAL SUMMARY =====")
    cat_cols = get_categorical_columns(df)
    for col in cat_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts())

def missing_value_report(df: pd.DataFrame):
    """Display number of missing values per column and a heatmap."""
    print("===== MISSING VALUES =====")
    missing = df.isnull().sum()
    display(missing)  # noqa: F821
    
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

# -----------------------------
# Correlation Analysis
# -----------------------------
def top_correlations(df: pd.DataFrame, target: str, n=5):
    """Display top N numeric features most correlated with a numeric target."""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if target not in numeric_df.columns:
        raise ValueError(f"Target '{target}' must be numeric and present in DataFrame.")
    top_corrs = numeric_df.corr()[target].abs().sort_values(ascending=False)
    top_corrs = top_corrs.drop(target).head(n)
    
    print(f"===== TOP {n} CORRELATIONS WITH '{target}' =====")
    display(top_corrs)  # noqa: F821
    return top_corrs

def plot_correlation_matrix(df):
    """Plot correlation heatmap for numeric columns."""
    numeric_cols = get_numeric_columns(df)
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

# -----------------------------
# Numeric Visualization
# -----------------------------
def plot_numeric_histograms(df):
    """Plot histograms for all numeric columns."""
    numeric_cols = get_numeric_columns(df)
    df[numeric_cols].hist(bins=30, figsize=(15,10), color='skyblue', edgecolor='black')
    plt.suptitle('Histograms of Numerical Features', fontsize=16)
    plt.show()

def plot_numeric_density(df):
    """Plot density plots for all numeric columns."""
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        sns.kdeplot(df[col], fill=True)
        plt.title(f'Density Plot of {col}')
        plt.show()

def plot_numeric_boxplots(df):
    """Boxplots for numeric columns to detect outliers."""
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

# -----------------------------
# Categorical Visualization
# -----------------------------
def plot_categorical_distribution(df, top_n=10):
    """Plot top N categories for all categorical columns to handle high-cardinality safely."""
    cat_cols = get_categorical_columns(df)
    if not cat_cols:
        print("No categorical columns to plot.")
        return

    for col in cat_cols:
        # Plot only top N categories to avoid huge plots
        top_values = df[col].value_counts().nlargest(top_n)
        plt.figure(figsize=(8,4))
        sns.barplot(x=top_values.values, y=top_values.index, palette='viridis')
        plt.title(f'Top {top_n} categories in {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.show()
# -----------------------------
# Outlier Detection
# -----------------------------
def detect_outliers(df: pd.DataFrame):
    """Detect outliers using IQR method and display summary."""
    numeric_cols = get_numeric_columns(df)
    print("===== OUTLIER DETECTION =====")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col}: {len(outliers)} outliers")
    plot_numeric_boxplots(df)

# -----------------------------
# Utility Functions
# -----------------------------
def get_numeric_columns(df):
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def get_categorical_columns(df):
    return df.select_dtypes(include='object').columns.tolist()
main


