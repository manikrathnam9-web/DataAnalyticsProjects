"""Export all Phase 3 EDA charts to images/ (non-interactive)."""
import matplotlib

matplotlib.use("Agg")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from eda_helpers import plot_attrition_rate, plot_ordinal_attrition

IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "hr_cleaned.csv")
sns.set_theme(style="whitegrid", palette="muted")

# 1
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df["Attrition"].value_counts()
axes[0].bar(counts.index, counts.values, color=["#2ecc71", "#e74c3c"])
axes[1].pie(counts, labels=counts.index, colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%")
plt.savefig(IMAGES_DIR / "eda_01_attrition_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

# Rate charts
for col, fname, horiz in [
    ("Department", "eda_02_department.png", False),
    ("JobRole", "eda_03_jobrole.png", True),
    ("Salary_Band", "eda_05_salary_band.png", False),
    ("Gender", "eda_09_gender.png", False),
    ("EducationField", "eda_10_education_field.png", True),
    ("BusinessTravel", "eda_18_business_travel.png", False),
    ("MaritalStatus", "eda_19_marital_status.png", False),
    ("OverTime", "eda_20_overtime.png", False),
]:
    plot_attrition_rate(df, col, col, horizontal=horiz, save_path=IMAGES_DIR / fname)
    plt.close()

# Income
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", hue="Attrition", legend=False, palette=["#2ecc71", "#e74c3c"], ax=axes[0])
sns.kdeplot(data=df, x="MonthlyIncome", hue="Attrition", fill=True, ax=axes[1])
plt.savefig(IMAGES_DIR / "eda_04_monthly_income.png", dpi=120, bbox_inches="tight")
plt.close()

# Age
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["Age"], bins=20, kde=True, ax=axes[0])
sns.histplot(data=df, x="Age", hue="Attrition", bins=20, kde=True, palette=["#2ecc71", "#e74c3c"], ax=axes[1])
plt.savefig(IMAGES_DIR / "eda_06_age_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

df["Age_Group"] = pd.cut(df["Age"], bins=[0, 30, 40, 50, 100], labels=["<=30", "31-40", "41-50", "50+"])
plot_attrition_rate(df, "Age_Group", "Age", save_path=IMAGES_DIR / "eda_07_age_attrition.png")
plt.close()

for col, fname, labels in [
    ("JobSatisfaction", "eda_11_job_satisfaction.png", {1: "1", 2: "2", 3: "3", 4: "4"}),
    ("EnvironmentSatisfaction", "eda_12_environment_satisfaction.png", {1: "1", 2: "2", 3: "3", 4: "4"}),
    ("RelationshipSatisfaction", "eda_13_relationship_satisfaction.png", {1: "1", 2: "2", 3: "3", 4: "4"}),
    ("StockOptionLevel", "eda_21_stock_options.png", {0: "0", 1: "1", 2: "2", 3: "3"}),
    ("TrainingTimesLastYear", "eda_22_training.png", None),
]:
    plot_ordinal_attrition(df, col, col, labels=labels, save_path=IMAGES_DIR / fname)
    plt.close()

df["Promotion_Category"] = pd.cut(df["YearsSinceLastPromotion"], bins=[-1, 1, 3, 6, 20], labels=["0-1", "2-3", "4-6", "7+"])
plot_attrition_rate(df, "Promotion_Category", "Promotion", save_path=IMAGES_DIR / "eda_14_promotion.png")
plt.close()

df["Tenure_Category"] = pd.cut(df["YearsAtCompany"], bins=[-1, 2, 5, 10, 100], labels=["0-2", "3-5", "6-10", "10+"])
plot_attrition_rate(df, "Tenure_Category", "Tenure", save_path=IMAGES_DIR / "eda_15_tenure.png")
plt.close()

fig, ax = plt.subplots()
sns.boxplot(data=df, x="Attrition", y="YearsSinceLastPromotion", hue="Attrition", legend=False, ax=ax)
plt.savefig(IMAGES_DIR / "eda_16_years_since_promotion.png", dpi=120, bbox_inches="tight")
plt.close()

df["Distance_Band"] = pd.cut(df["DistanceFromHome"], bins=[0, 5, 10, 20, 30], labels=["1-5", "6-10", "11-20", "21+"])
plot_attrition_rate(df, "Distance_Band", "Distance", save_path=IMAGES_DIR / "eda_17_distance.png")
plt.close()

numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Attrition_Flag", "Any_Outlier_Flag"]]
corr = df[numeric_cols + ["Attrition_Flag"]].corr()
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=False, cmap="RdBu_r", center=0, ax=ax)
plt.savefig(IMAGES_DIR / "eda_23_correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

print(f"Exported {len(list(IMAGES_DIR.glob('eda_*.png')))} charts to {IMAGES_DIR}")
