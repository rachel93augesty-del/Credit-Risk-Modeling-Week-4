# src/train.py

import pandas as pd
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from data_processing import load_data, build_preprocessing_pipeline

# -----------------------------
# 1. Load & preprocess data
# -----------------------------
def prepare_data(path: str, target_col='Target'):
    """
    Load CSV data, preprocess, and split into train/test sets.
    """
    df = load_data(path)
    numeric_features = ['Amount', 'Value']  # adjust as needed
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory', 'ChannelId']
    
    pipeline = build_preprocessing_pipeline(numeric_features, categorical_features)
    
    X = pipeline.fit_transform(df.drop(columns=[target_col]))
    y = df[target_col]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 2. Define models & hyperparams
# -----------------------------
def get_models():
    """
    Returns a dictionary of models and their hyperparameter grids.
    """
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
        "DecisionTree": (DecisionTreeClassifier(), {"max_depth": [3, 5, 10, None]}),
        "RandomForest": (RandomForestClassifier(), {"n_estimators": [50, 100], "max_depth": [5, 10]}),
        "GradientBoosting": (GradientBoostingClassifier(), {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]})
    }
    return models

# -----------------------------
# 3. Evaluate model
# -----------------------------
def evaluate_model(model, X_test, y_test, plot_roc=True):
    """
    Returns a dictionary of evaluation metrics and optionally plots ROC curve.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Plot ROC curve
    if plot_roc and y_prob is not None:
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.show()
    
    return metrics

# -----------------------------
# 4. Train & track models
# -----------------------------
def train_and_track_models(X_train, X_test, y_train, y_test):
    """
    Trains all models, logs metrics to MLflow, and registers the best model.
    """
    mlflow.set_experiment("Credit_Risk_Models")
    
    best_model_name = None
    best_score = 0
    best_model_obj = None
    
    for name, (model, params) in get_models().items():
        print(f"Training {name}...")
        
        # Hyperparameter tuning
        gs = GridSearchCV(model, params, cv=3, scoring='roc_auc')
        gs.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(gs.best_estimator_, X_test, y_test, plot_roc=False)
        
        # MLflow logging
        with mlflow.start_run(run_name=name):
            mlflow.log_params(gs.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(gs.best_estimator_, artifact_path="model", registered_model_name=name)
        
        print(f"{name} metrics:", metrics)
        
        # Track best model
        if metrics["roc_auc"] and metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model_name = name
            best_model_obj = gs.best_estimator_
    
    print(f"Best model: {best_model_name} with ROC-AUC: {best_score}")
    return best_model_name, best_model_obj

# -----------------------------
# 5. Main execution
# -----------------------------
if __name__ == "__main__":
    # Load and split data
    X_train, X_test, y_train, y_test = prepare_data("data/processed/rfm_model_ready.csv", target_col="HighRisk")
    
    # Train models and track with MLflow
    best_model_name, best_model = train_and_track_models(X_train, X_test, y_train, y_test)
    
    print(f"Task 5 complete! Best model: {best_model_name}")
