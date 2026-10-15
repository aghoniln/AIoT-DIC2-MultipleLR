# 🚀 MLR Studio: Multiple Linear Regression & Feature Selection Explorer (DIC-2)

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?logo=github&style=flat-square)](https://aghoniln.github.io/AIoT_L5_CWA_HW1/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&style=flat-square)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange?logo=scikitlearn&style=flat-square)](https://scikit-learn.org/)
[![Framework](https://img.shields.io/badge/Methodology-CRISP--DM-purple?style=flat-square)](#crisp-dm-standardized-workflow)

> **AIoT Course Homework (DIC-2)**: Multiple Linear Regression & Parsimony Optimization on **Kaggle 50 Startups** and **Boston Housing** datasets.  
> **Course Requirement Fulfillment**: *"Do two problems but choose one of the topics to upload; the second problem directly applies the standardized skill / workflow."*

---

## 🌐 Live Interactive Application

👉 **[https://aghoniln.github.io/AIoT_L5_CWA_HW1/](https://aghoniln.github.io/AIoT_L5_CWA_HW1/)**

An interactive, client-side analytics studio built with **Vanilla HTML/CSS/JavaScript** and **Chart.js**, featuring live what-if simulation, regression diagnostics, and real-time model comparison.

---

## 🎯 Dual-Problem Architecture

This repository demonstrates the power of a **reusable Machine Learning workflow**:
1. **Problem 1 (Primary Topic Upload)**: **50 Startups Venture Profit Optimization**
   - Predicts venture profit based on R&D Spend, Administration, Marketing Spend, and State location (California, New York, Florida).
   - Identifies the parsimonious **optimal sweet spot ($k=2$)** vs. the **overfitting cliff ($k=5$)**.
2. **Problem 2 (Workflow Transfer)**: **Boston Housing Property Valuation**
   - Applies the **identical automated feature selection & evaluation pipeline** to predict Median Home Values (`medv` in $1,000s) from socio-economic and structural census variables (`rm`, `lstat`, `ptratio`, `dis`, `crim`).
   - Switchable with one click on the live web dashboard!

---

## 🔬 Key Experimental Findings

### 50 Startups Problem (Holdout Test Set: 20%)

| Feature Count ($k$) | Selected Predictors | Test RMSE ($) | Test $R^2$ | Model Evaluation |
| :---: | :--- | :---: | :---: | :--- |
| **$k = 1$** | R&D Spend | $8,274.87 | 0.9465 | Strong baseline predictor |
| **$k = 2$ ★** | **R&D Spend + Marketing Spend** | **$8,198.80** | **0.9474** | **★ Global Sweet Spot (Min Error, Peak $R^2$)** |
| **$k = 3$** | R&D + Marketing + State (NY) | $8,309.06 | 0.9460 | Slight generalization penalty |
| **$k = 4$** | R&D + Marketing + State (NY, FL) | $8,409.92 | 0.9447 | Minor variance penalty |
| **$k = 5$ ⚠** | R&D + Marketing + State (NY, FL) + Admin | **$9,137.99** | **0.9347** | **⚠ Overfitting Cliff (+$939.19 RMSE)** |

> **Occam's Razor Insight**: Including `Administration` and categorical `State` dummy variables creates unnecessary model complexity that deteriorates out-of-sample generalization. The **2-feature model** achieves the highest out-of-sample prediction accuracy.

---

## ⚙ 5 Feature Selection Algorithms Compared

1. **Lasso (L1 Regularization)**: Embedded path regularization shrinking redundant weights to zero as $\alpha$ increases.
2. **Forward SFS (Sequential Feature Selection)**: Greedy step-up algorithm iteratively adding the feature minimizing test RMSE.
3. **RFE (Recursive Feature Elimination)**: Backward wrapper pruning the least significant weights.
4. **ElasticNet**: Hybrid $L_1 + L_2$ regularized ranking balancing sparsity and collinear grouping.
5. **Exhaustive Combinatorial Search**: Evaluates all $\sum \binom{p}{k}$ feature subsets to guarantee the mathematical global minimum.

---

## 📊 Statistical Inference & Diagnostics

For every fitted model, the system computes:
- **Parameter Estimates ($\beta$)** and **Standard Errors**
- **$t$-Statistics** and **Two-tailed $p$-Values** (with significance markers: $p < 0.001^{***}$, $p < 0.05^{*}$)
- **95% Confidence Intervals** $[CI_{lower}, CI_{upper}]$
- **Variance Inflation Factor (VIF)** to monitor and prevent multicollinearity:
  $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
  *(All $k=2$ predictors maintain $\text{VIF} < 2.5$, safely below the collinear threshold of 5.0)*

---

## 🔄 CRISP-DM Standardized Workflow

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│ 1. Business Understand │ ───> │ 2. Data Understanding  │ ───> │  3. Data Preparation   │
│ Venture Profit / MEDV  │      │ Correlation & Missing  │      │ Encoding, Scaler, Split│
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
                                                                             │
┌────────────────────────┐      ┌────────────────────────┐                   ▼
│     6. Deployment      │ <─── │     5. Evaluation      │ <─── ┌────────────────────────┐
│ GitHub Pages Web App   │      │ Parsimony vs Overfit   │      │      4. Modeling       │
│ & Joblib Model Export  │      │ Test Residuals & VIF   │      │ 5 Selection Algorithms │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

---

## 📁 Repository File Structure

```text
├── 50_Startups.csv                        # Raw 50 Startups dataset
├── BostonHousing.csv                      # Raw Boston Housing dataset
├── 50_Startup_Problem_Lasso_Complete.ipynb # Jupyter Notebook solution
├── compute_results.py                     # Dual-dataset precomputation pipeline
├── crisp_dm_startup.py                    # End-to-end CRISP-DM training script
├── part1_mlr.py                           # Baseline OLS Multiple Linear Regression
├── part2_forward_selection.py             # Forward SFS implementation
├── part3_lasso.py                         # Lasso path implementation
├── startup_profit_model.joblib            # Serialized production model
├── metrics.json                           # Evaluated metrics record
├── index.html                             # Web application (served by GitHub Pages)
├── style.css                              # Obsidian Dark UI theme & design system
├── app.js                                 # Application logic & interactive simulator
├── data.js                                # Precomputed results bundle
└── assets/                                # Figures and datasets
    ├── 50_Startups.csv
    ├── BostonHousing.csv
    ├── actual_vs_predicted.png
    └── forward_selection.png
```

---

## 💻 Local Quickstart

### 1. Run Web Application Locally
No build tools or Node.js required! Simply start any HTTP server:

```bash
# Python 3
python -m http.server 8000

# Open in browser:
http://localhost:8000
```

### 2. Regenerate Results Pipeline
To re-run the scikit-learn feature-selection and inference computation for both datasets:

```bash
python compute_results.py
```

---

## 👤 Author & Course Information

- **Student GitHub**: [@aghoniln](https://github.com/aghoniln)
- **Repository**: [aghoniln/AIoT_L5_CWA_HW1](https://github.com/aghoniln/AIoT_L5_CWA_HW1)
- **Course**: AIoT (Artificial Intelligence of Things) — DIC-2 Multiple Linear Regression Assignment
