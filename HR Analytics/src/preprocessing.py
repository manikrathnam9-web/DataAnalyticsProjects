"""
HR Analytics – reusable preprocessing utilities.

Used by notebooks and (optionally) app.py for consistent data preparation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Columns with no predictive value (constant or identifier)
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

# Nominal categoricals — one-hot encoded for ML
NOMINAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "Salary_Band",
    "Age_Group",
    "Experience_Group",
    "Tenure_Category",
    "Promotion_Category",
]

# Ordinal numerics — kept as integers (Likert / level scales)
ORDINAL_COLS = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
]

TARGET_COL = "Attrition"


def load_raw_data(data_path: Path | str) -> pd.DataFrame:
    """Load the IBM HR attrition CSV."""
    return pd.read_csv(data_path)


def treat_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values when present.

    This dataset has no missing values; the function documents the strategy
    we would apply in production HR systems.
    """
    df = df.copy()
    if df.isnull().sum().sum() == 0:
        return df

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and duplicate employee IDs."""
    df = df.copy()
    df = df.drop_duplicates()
    if "EmployeeNumber" in df.columns:
        df = df.drop_duplicates(subset=["EmployeeNumber"], keep="first")
    return df.reset_index(drop=True)


def drop_non_predictive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove constant fields and employee identifier."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def create_salary_band(
    df: pd.DataFrame,
    income_col: str = "MonthlyIncome",
    band_col: str = "Salary_Band",
) -> pd.DataFrame:
    """
    Quartile-based salary bands aligned to this dataset's income distribution.

    Uses qcut so band boundaries adapt to the data rather than arbitrary cutoffs.
    """
    df = df.copy()
    df[band_col] = pd.qcut(
        df[income_col],
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )
    return df


def create_experience_group(
    df: pd.DataFrame,
    years_col: str = "TotalWorkingYears",
    group_col: str = "Experience_Group",
) -> pd.DataFrame:
    """
    Career experience segments based on total working years.

    Bins reflect typical HR career stages in this workforce sample.
    """
    df = df.copy()
    df[group_col] = pd.cut(
        df[years_col],
        bins=[-1, 5, 10, 20, 100],
        labels=["Early Career (0-5)", "Mid Career (6-10)", "Experienced (11-20)", "Veteran (20+)"],
    )
    return df


def create_age_group(
    df: pd.DataFrame,
    age_col: str = "Age",
    group_col: str = "Age_Group",
) -> pd.DataFrame:
    """
    Life-stage age groups based on typical career phases.
    """
    df = df.copy()
    df[group_col] = pd.cut(
        df[age_col],
        bins=[-1, 29, 39, 49, 120],
        labels=["Under 30", "30-39", "40-49", "50+"],
    )
    return df


def create_tenure_category(
    df: pd.DataFrame,
    tenure_col: str = "YearsAtCompany",
    category_col: str = "Tenure_Category",
) -> pd.DataFrame:
    """
    Organizational tenure categories.
    """
    df = df.copy()
    df[category_col] = pd.cut(
        df[tenure_col],
        bins=[-1, 2, 5, 10, 100],
        labels=["Newbie (0-2)", "Junior (3-5)", "Mid-Level (6-10)", "Senior (10+)"],
    )
    return df


def create_promotion_category(
    df: pd.DataFrame,
    promo_col: str = "YearsSinceLastPromotion",
    category_col: str = "Promotion_Category",
) -> pd.DataFrame:
    """
    Stagnation monitoring based on years since last promotion.
    """
    df = df.copy()
    df[category_col] = pd.cut(
        df[promo_col],
        bins=[-1, 1, 5, 100],
        labels=["Recent Promotion (0-1)", "Mid-tenure (2-5)", "Stagnant (6+)"],
    )
    return df


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Iterable[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Flag IQR-based outliers per column. Does not remove rows — long tenure
    and high earners are often legitimate retention signals, not errors.
    """
    flags = pd.DataFrame(index=df.index)
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        flags[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)
    flags["any_outlier"] = flags.any(axis=1)
    return flags


def encode_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Binary-encode attrition: Yes=1 (left), No=0 (stayed)."""
    df = df.copy()
    df["Attrition_Flag"] = (df[target_col] == "Yes").astype(int)
    return df


def encode_features_onehot(
    df: pd.DataFrame,
    nominal_cols: list[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode nominal categoricals for ML-ready feature matrix.

    Ordinal and numeric columns are retained as-is.
    """
    nominal_cols = nominal_cols or NOMINAL_COLS
    present = [c for c in nominal_cols if c in df.columns]
    return pd.get_dummies(df, columns=present, drop_first=drop_first, dtype=int)


def build_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for EDA and dashboard use."""
    df = treat_missing_values(df)
    df = remove_duplicates(df)
    df = drop_non_predictive_columns(df)
    df = create_salary_band(df)
    df = create_experience_group(df)
    df = create_age_group(df)
    df = create_tenure_category(df)
    df = create_promotion_category(df)
    df = encode_target(df)
    return df


def build_ml_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Produce one-hot encoded dataset excluding raw target text column."""
    feature_df = df_clean.drop(columns=[TARGET_COL])
    return encode_features_onehot(feature_df)
