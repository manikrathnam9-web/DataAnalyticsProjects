"""
HR Analytics – Machine Learning Utilities.

Contains functions for train-test splitting, preprocessing pipelines,
hyperparameter tuning, model evaluation, and plotting model performance.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def load_ml_data(filepath: Path | str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load ML ready data and split into features (X) and target (y).
    Expects Attrition_Flag as the binary target column.
    """
    df = pd.read_csv(filepath)
    if "Attrition_Flag" not in df.columns:
        raise ValueError("Target column 'Attrition_Flag' not found in dataset.")
    
    X = df.drop(columns=["Attrition_Flag"])
    y = df["Attrition_Flag"]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into train and test sets using stratified sampling
    to preserve the imbalance ratio of the target.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Apply standard scaling to continuous numeric features.
    Saves scaler for production pipeline.
    """
    scaler = StandardScaler()
    
    # Identify continuous columns (not dummy-encoded or binary)
    # We can identify them by checking which columns have values outside {0, 1}
    # Or by checking numeric columns that are not columns containing underscores (dummy encoded)
    # Let's scale all columns that are not binary (i.e. have max > 1 or min < 0)
    cols_to_scale = [
        col for col in X_train.columns
        if not (
            X_train[col].nunique() == 2 and 
            set(X_train[col].unique()).issubset({0, 1, 0.0, 1.0})
        )
    ]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if cols_to_scale:
        X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
        
    return X_train_scaled, X_test_scaled, scaler


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    """Compute standard classification evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = 0.5
    return metrics


def plot_evaluation_curves(
    y_test: pd.Series,
    y_prob: np.ndarray,
    y_pred: pd.Series,
    model_name: str,
    save_dir: Path | str,
) -> None:
    """Generate and save Confusion Matrix, ROC Curve, and PR Curve."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    clean_name = model_name.lower().replace(" ", "_")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed (0)", "Left (1)"],
        yticklabels=["Stayed (0)", "Left (1)"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    plt.savefig(save_path / f"{clean_name}_confusion_matrix.png", dpi=120)
    plt.close()
    
    # 2. ROC & PR Curves in a single figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve – {model_name}")
    axes[0].legend(loc="lower right")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recall, precision, color="blue", lw=2, label="PR Curve")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve – {model_name}")
    axes[1].legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(save_path / f"{clean_name}_curves.png", dpi=120)
    plt.close()


def tune_model(
    model: Any,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    scoring: str = "f1",
) -> Any:
    """Tune hyperparameters using GridSearchCV with Stratified K-Fold."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def save_model_artifact(
    model: Any,
    scaler: StandardScaler | None,
    feature_names: list[str],
    model_name: str,
    save_dir: Path | str,
) -> None:
    """Save trained model, scaler, and feature names as a unified joblib dictionary."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names
    }
    
    clean_name = model_name.lower().replace(" ", "_")
    joblib.dump(artifact, save_path / f"{clean_name}_artifact.joblib")
    print(f"Saved {model_name} artifact to {save_path / f'{clean_name}_artifact.joblib'}")
