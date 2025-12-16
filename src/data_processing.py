# src/data_processing.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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
    print(df.describe())

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
    print(missing)
    
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
    print(top_corrs)
    return top_corrs

def plot_correlation_matrix(df: pd.DataFrame):
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
def plot_numeric_histograms(df: pd.DataFrame):
    numeric_cols = get_numeric_columns(df)
    df[numeric_cols].hist(bins=30, figsize=(15,10), color='skyblue', edgecolor='black')
    plt.suptitle('Histograms of Numerical Features', fontsize=16)
    plt.show()

def plot_numeric_density(df: pd.DataFrame):
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        sns.kdeplot(df[col], fill=True)
        plt.title(f'Density Plot of {col}')
        plt.show()

def plot_numeric_boxplots(df: pd.DataFrame):
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

# -----------------------------
# Categorical Visualization
# -----------------------------
def plot_categorical_distribution(df: pd.DataFrame, top_n=10):
    cat_cols = get_categorical_columns(df)
    if not cat_cols:
        print("No categorical columns to plot.")
        return

    for col in cat_cols:
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
# Train/Test Split (Task 5)
# -----------------------------
def split_data(df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    """Split dataframe into train and test sets with stratification."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

# -----------------------------
# Utility Functions
# -----------------------------
def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def get_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(include=['object']).columns.tolist()
