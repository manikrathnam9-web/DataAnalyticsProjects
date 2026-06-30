"""
HR Analytics – SHAP Explainability Utilities.

Provides functions to compute SHAP values and generate SHAP plots
(summary plot, bar plot, waterfall plot, force plot) to interpret model predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def load_model_and_data(
    model_path: Path | str,
    data_path: Path | str,
) -> tuple[Any, pd.DataFrame]:
    """Load the trained model artifact and ML-ready features (X)."""
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    
    df = pd.read_csv(data_path)
    if "Attrition_Flag" in df.columns:
        df = df.drop(columns=["Attrition_Flag"])
    
    # Align features with training names
    X = df[feature_names]
    return model, X


def compute_shap_values(model: Any, X: pd.DataFrame) -> tuple[shap.Explanation, shap.TreeExplainer]:
    """Compute SHAP values using TreeExplainer for the tree-based model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values, explainer


def generate_shap_plots(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    save_dir: Path | str,
) -> None:
    """Generate and save SHAP summary, bar, waterfall, and force plots."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. SHAP Summary Plot (density plot)
    plt.figure(figsize=(10, 8))
    # Note: shap.summary_plot works on tree explanations.
    # For shap_values object, we index class 1 (attrition) if multiclass,
    # but for binary Random Forest in scikit-learn, shap_values might have shape (N, D, 2).
    # TreeExplainer on RandomForestClassifier returns SHAP values for both classes (0 and 1).
    # We explain class 1 (attrition).
    if len(shap_values.shape) == 3:
        # multiclass shape: (samples, features, classes)
        shap_exp = shap_values[:, :, 1]
    else:
        shap_exp = shap_values
        
    shap.summary_plot(shap_exp, X, show=False)
    plt.title("SHAP Summary Plot (Feature Attrition Impact)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path / "shap_summary_plot.png", dpi=120, bbox_inches="tight")
    plt.close()
    
    # 2. SHAP Bar Plot (global importance)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_exp, show=False)
    plt.title("SHAP Feature Importance (Mean Absolute Impact)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path / "shap_bar_plot.png", dpi=120, bbox_inches="tight")
    plt.close()
    
    # 3. Local Explainability: Waterfall Plot for a high-risk employee
    # Find an employee index where the model predicts high attrition probability
    # For simplicity, we will save the waterfall plot for index 0 (or a specific high-risk index)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_exp[0], show=False)
    plt.title("SHAP Waterfall Plot – Individual Employee Prediction (Index 0)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path / "shap_waterfall_plot.png", dpi=120, bbox_inches="tight")
    plt.close()
    
    # 4. Local Explainability: Force Plot for index 0
    # shap.force_plot outputs HTML. We can save the raw values or plot it.
    # To write it to file, we render it to HTML
    try:
        # Get expected value for class 1
        expected_value = shap_exp.base_values
        if isinstance(expected_value, np.ndarray) and len(expected_value) > 1:
            expected_value = expected_value[0]
            
        force_plot_html = shap.force_plot(
            expected_value,
            shap_exp.values[0],
            X.iloc[0],
            matplotlib=False
        )
        shap.save_html(str(save_path / "shap_force_plot.html"), force_plot_html)
        print("Saved SHAP Force Plot HTML.")
    except Exception as e:
        print(f"Skipped SHAP Force Plot HTML generation due to: {e}")
        
    print(f"Generated all SHAP plots in {save_path}")
