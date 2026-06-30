# Power BI HR Analytics Attrition Dashboard Guide

This document serves as the implementation blueprint for the professional Power BI dashboard. It details the design system, layout structure for all 3 pages, and the copy-pasteable DAX formulas required to build the metrics.

---

## 1. Design System & Theme Settings

To ensure the dashboard looks cohesive and premium, configure the following styling options in Power BI under **View > Themes > Customize Current Theme**:

*   **Color Palette (Modern Cool Tech):**
    *   **Primary Accent:** `#1E293B` (Deep Slate / background elements)
    *   **Secondary Accent:** `#3B82F6` (Electric Blue / active items)
    *   **Alert / Exit State:** `#EF4444` (Vibrant Coral / attrition / exit signal)
    *   **Success / Retained State:** `#10B981` (Emerald Green / retained employees)
    *   **Neutral Dark:** `#0F172A` (Slate Dark / text and headers)
    *   **Neutral Light:** `#F8FAFC` (Cool Off-White / page background)
*   **Typography:**
    *   **Titles & Metrics:** `Segoe UI Semibold` or `Segoe UI Bold`
    *   **Labels & Body Text:** `Segoe UI` or `Din` (for numbers)
*   **Visual Container Settings:**
    *   **Border Radius:** `8px` rounded corners
    *   **Shadow:** Subtle bottom-right shadow (Color: `#000000`, Transparency: `90%`, Size: `4px`, Blur: `4px`)
    *   **Padding:** Standard `12px` inner margins

---

## 2. Page 1: Executive Summary (High-Level Overview)

### Visual Layout Grid (3x3 grid layout)
```
+-----------------------------------------------------------------------------+
|                                  HEADER SECTION                             |
|  [Logo]  HR Analytics Attrition Executive Summary    [Slicers: Dept, Job, G]|
+-----------------------------------------------------------------------------+
|                                  KPI CARDS                                  |
| [Total Headcount] [Exited Employees] [Attrition Rate] [Avg Income] [Avg Age]|
+-----------------------------------------------------------------------------+
|  LEFT: Headcount Trends by Dept       |  RIGHT: Attrition Rate by Department |
|  (Stacked Bar: Stayed vs Left)        |  (Horizontal Bar Chart - Sort Desc)  |
+-----------------------------------------------------------------------------+
|  BOTTOM LEFT: Job Role Matrix         |  BOTTOM RIGHT: Key Attrition Drivers|
|  (Table with conditional formatting)  |  (Donut: Attrition by Overtime Status|
+-----------------------------------------------------------------------------+
```

### Visual Specifications
1.  **KPI Cards (New Multi-Row Card Visual):**
    *   **Total Headcount:** Count of employees (`Total Employees`).
    *   **Employees Left:** Count of employees where attrition is active (`Employees Left`).
    *   **Attrition Rate:** Percentage of headcount lost (`Attrition Rate`).
    *   **Average Salary:** Mean monthly salary (`Average Salary`).
    *   **Average Age:** Mean age (`Average Age`).
    *   **Avg Years at Company:** Mean company tenure (`Average Years at Company`).
2.  **Top Right Filters (Slicer Visuals):**
    *   **Department:** Dropdown format, single-select off.
    *   **Job Role:** Searchable list format.
    *   **Gender:** Horizontal tile format.
3.  **Headcount by Department (Stacked Bar Chart):**
    *   **X-axis:** Department
    *   **Y-axis:** Count of Employees
    *   **Legend:** Attrition Status (Stayed = `#10B981`, Left = `#EF4444`)
4.  **Attrition Rate by Job Role (Clustered Bar Chart):**
    *   **Y-axis:** Job Role
    *   **X-axis:** `Attrition Rate` (DAX Measure)
    *   **Target Line:** Add constant line at `16.1%` (Company Average) in gray dashed formatting.

---

## 3. Page 2: Attrition Analysis (Deep-Dive Analysis)

### Visual Layout Grid
```
+-----------------------------------------------------------------------------+
| [Back to Exec]            HR Attrition Deep-Dive Analysis                   |
+-----------------------------------------------------------------------------+
|  TOP LEFT: Attrition by Age Group     |  TOP RIGHT: Attrition by Salary Band|
|  (Line / Area Chart)                  |  (Clustered Column Chart)           |
+-----------------------------------------------------------------------------+
|  MID LEFT: Overtime & Commute         |  MID RIGHT: Promotion Lag influence |
|  (Scatter: Dist. vs Age by Attrition) |  (Stacked Column: Years since Promo)|
+-----------------------------------------------------------------------------+
|  BOTTOM SECTION: Satisfaction Heatmap Matrix                                |
|  (Columns: Environment Satisf., Rows: Job Satisf., Cell color: Attr. Rate)  |
+-----------------------------------------------------------------------------+
```

### Visual Specifications
1.  **Attrition by Age Group (Clustered Column Chart):**
    *   **X-axis:** `Age_Group` (Under 30, 30-39, 40-49, 50+)
    *   **Y-axis:** `Attrition Rate` (DAX measure)
    *   *Highlight:* Visual shows the massive spike (~29%) in the `Under 30` age segment.
2.  **Attrition by Salary Band (Clustered Column Chart):**
    *   **X-axis:** `Salary_Band` (Low, Medium, High, Very High)
    *   **Y-axis:** `Attrition Rate`
    *   *Color:* Gradient scale ranging from Emerald (`Very High` band) to Coral Red (`Low` band).
3.  **Overtime Impact (Donut Chart):**
    *   **Legend:** OverTime (Yes vs No)
    *   **Values:** `Employees Left` (DAX measure)
    *   *Insight:* Shows that overtime workers comprise a disproportionate share of exits.
4.  **Job Satisfaction Heatmap (Matrix Visual):**
    *   **Rows:** `JobSatisfaction` (1 = Low, 2 = Medium, 3 = High, 4 = Very High)
    *   **Columns:** `EnvironmentSatisfaction` (1, 2, 3, 4)
    *   **Values:** `Attrition Rate`
    *   **Conditional Formatting:** Background color gradient using a diverging palette ( Emerald -> Yellow -> Red) for the cell values to highlight low satisfaction grid cells.

---

## 4. Page 3: Prediction Insights (ML & Actionable Dashboard)

### Visual Layout Grid
```
+-----------------------------------------------------------------------------+
| [Overview]              Predictive Attrition Risk Portal                    |
+-----------------------------------------------------------------------------+
|  LEFT: Top Predictive Drivers         |  RIGHT: High-Risk Employee Profile   |
|  (Clustered Horizontal Bar Chart)     |  (Table: Detail list of active staff |
|                                       |   with Attrition Probability > 60%)  |
+-----------------------------------------------------------------------------+
|  BOTTOM LEFT: Commute Fatigue Impact  |  BOTTOM RIGHT: Strategic Actions    |
|  (Commute bin vs Attrition Rate)      |  (Text Card: Recommendations list)  |
+-----------------------------------------------------------------------------+
```

### Visual Specifications
1.  **Feature Importance (Clustered Bar Chart):**
    *   **Y-axis:** Feature Name
    *   **X-axis:** Random Forest Importance Score (generated in Phase 5 and imported).
2.  **High-Risk Employee Profile Table:**
    *   **Columns:** `Employee ID`, `Job Role`, `Department`, `Monthly Income`, `Overtime`, `Age`, `Predicted Attrition Probability`.
    *   **Filter on Visual:** `Predicted Attrition Probability >= 60%` AND `Attrition = No` (Active employees).
    *   *Sorting:* Sort descending by probability. This serves as HR's "Action Watchlist".
3.  **commute Distance Category (Column Chart):**
    *   **X-axis:** Commute Distance Bins (0-5 miles, 6-15 miles, 16+ miles)
    *   **Y-axis:** `Attrition Rate`

---

## 5. Required DAX Measures

Copy and paste these formulas into the Power BI modeling window:

### 1. Total Employees (Headcount)
```dax
Total Employees = COUNTROWS('hr_cleaned')
```

### 2. Employees Left
```dax
Employees Left = CALCULATE(
    COUNTROWS('hr_cleaned'),
    'hr_cleaned'[Attrition] = "Yes"
)
```

### 3. Attrition Rate (%)
```dax
Attrition Rate = 
DIVIDE(
    [Employees Left],
    [Total Employees],
    0
) * 100
```

### 4. Average Salary ($)
```dax
Average Salary = AVERAGE('hr_cleaned'[MonthlyIncome])
```

### 5. Average Age
```dax
Average Age = AVERAGE('hr_cleaned'[Age])
```

### 6. Average Years at Company
```dax
Average Years at Company = AVERAGE('hr_cleaned'[YearsAtCompany])
```

### 7. Risk Segment Indicator (Conditional Metric)
```dax
Risk Segment = 
IF(
    [Attrition Rate] >= 25, 
    "🚨 CRITICAL (High Exit)", 
    IF(
        [Attrition Rate] >= 16, 
        "⚠️ WARNING (Elevated)", 
        "✅ STABLE (Low Turnover)"
    )
)
```
