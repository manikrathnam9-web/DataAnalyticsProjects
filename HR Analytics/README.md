# HR Analytics: Predict Employee Attrition & Retention Drivers

An end-to-end data science and machine learning project designed to predict voluntary employee attrition and diagnose key attrition drivers (such as overtime burnout and compensation gaps) using **Explainable AI (SHAP)**.

This repository features data preprocessing, feature engineering, classification benchmarking, explainability analyses, and a complete design guide for a Power BI dashboard. It is structured to serve as a portfolio-grade project for recruiter evaluation and data science interviews.

---

## Project Structure

The codebase is organized cleanly as follows:

```
HR-Analytics/
│
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv      # Raw dataset from IBM HRIS
│   └── processed/
│       ├── hr_cleaned.csv                         # Cleaned dataset (for EDA & dashboards)
│       └── hr_ml_ready.csv                        # One-hot encoded dataset (for ML models)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb                # Initial exploration and structural review
│   ├── 02_data_preprocessing.ipynb                # Duplicates, missing values, outlier treatment
│   ├── 03_exploratory_data_analysis.ipynb         # Bivariate & multivariate attrition analysis
│   ├── 04_feature_engineering.ipynb               # Binned variables & domain feature creation
│   ├── 05_machine_learning.ipynb                  # Training, cross-validation & model comparisons
│   └── 06_model_explainability.ipynb              # SHAP values & individual risk diagnosis
│
├── src/
│   ├── __init__.py
│   ├── eda_helpers.py                             # Reusable plotting modules for notebooks
│   ├── preprocessing.py                           # Preprocessing pipeline and feature engineers
│   ├── modeling.py                                # ML training and metric calculation helpers
│   └── explainability.py                          # SHAP explainability visualizers
│
├── models/                                         # Unified joblib model artifacts (stored here after execution)
│
├── dashboard/
│   └── dashboard_guide.md                         # Wireframes, layout grid, and copy-paste DAX formulas
│
├── reports/
│   ├── model_comparison.csv                       # Tabular comparison metrics of the ML models
│   ├── business_insights_recommendations.md       # Strategic analysis report and ranked programs
│   └── project_report.md                          # Professional academic-quality final report
│
├── images/                                         # Generated charts (distribution, curves, SHAP plots)
│
├── requirements.txt                                # Python library dependencies
└── README.md                                      # Repository documentation (this file)
```

---

## Objectives

1.  **Prediction:** Build multiple classification models to alert HR of high-risk active employees.
2.  **Explainability:** Decipher "black-box" predictions to identify *why* specific cohorts resign, mapping out exact feature directions (positive vs negative impact).
3.  **Actionable Retention:** Rank-order retention recommendations by business impact to optimize HR budget allocation.
4.  **Business Intelligence:** Provide DAX measures and layout schematics to build an executive-ready Power BI dashboard.

---

## Dataset

*   **Source:** IBM HR Employee Attrition and Performance (1,470 employee records, 35 attributes).
*   **Target:** `Attrition` (Yes = Left, No = Stayed).
*   **Imbalance:** Class imbalance of **16.1% Attrition Rate** (237 exits vs 1,233 retained), necessitating stratified splits and metrics beyond raw accuracy (Recall and F1).

---

## Technologies Used

*   **Core Logic & Preprocessing:** Python (Pandas, NumPy, Scikit-Learn)
*   **Visualizations:** Matplotlib, Seaborn, SHAP
*   **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)
*   **Business Intelligence:** Power BI (DAX)
*   **Notebook Environment:** Jupyter / Jupyterlab

---

## Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/HR-Analytics.git
    cd HR-Analytics
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Pipeline:**
    You can run the notebooks in sequence from `01_data_understanding.ipynb` to `06_model_explainability.ipynb` to regenerate the datasets, train the models, and calculate the SHAP explainability values.

---

## Results Summary

During Phase 5, three tuned classifiers were compared using Stratified 5-Fold Cross-Validation:

| Metric | Logistic Regression (Balanced) | Decision Tree (Tuned) | Random Forest (Best Fit) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 76.53% | 79.59% | **85.71%** |
| **Precision** | 38.64% | 40.54% | **61.90%** |
| **Recall** | **72.34%** | 31.91% | 55.32% |
| **F1 Score** | 50.37% | 35.71% | **58.43%** |
| **ROC-AUC** | 0.816 | 0.658 | **0.823** |

### Key Decision:
*   While Logistic Regression has a higher Recall, its Precision is very low, creating numerous false positives.
*   **Random Forest** is selected as the production model due to its high **ROC-AUC (0.823)**, superior generalizability, and the best balanced **F1 Score (58.43%)**.

---

## Actionable Business Recommendations (Summary)

1.  **Overtime Cap Policy:** Restrict consecutive overtime cycles to combat burnout. Employees working overtime show a **30.6% exit rate** compared to **10.4%** for standard hours.
2.  **Sales Rep Base Salary Adjustments:** Sales Representatives have a **39.8% exit rate**. Increase base pay and adjust commission thresholds to stabilize earnings.
3.  **Onboarding Peer Mentoring:** Establish peer buddies for employees during their first 24 months, where exit rates peak at **29.2%**.
4.  **Commute Mitigation:** Implement hybrid schedules or travel subsidies for employees commuting more than 15 miles (exit rate **22.4%**).

---

## Future Improvements

*   **Time-series Integration:** Analyze employee changes longitudinally (e.g., changes in satisfaction scores or promotions).
*   **Sentiment Analysis:** Integrate anonymous text surveys with structured database variables.
*   **Active Scoring API:** Wrap the trained Random Forest model in a FastAPI service to serve real-time employee risk scores directly to the HR dashboard.

---

## Author

*   **Jeshwanth**
*   Data Science & HR Analytics Portfolio Project
