# Dynamic Delivery Time Prediction System

A machine learning project that predicts e-commerce delivery times and identifies the factors that drive delivery delays, using the Olist Brazilian E-Commerce dataset.

**Author:** M M D H Malporu  
**Student ID:** IM/2023/015  
**Group:** Data Science Group 1  
**Affiliation:** Department of Industrial Management, University of Kelaniya  

---

## Project Overview

**Domain:** Business Analytics — E-Commerce Logistics & Last-Mile Delivery Optimization

**Problem:** E-commerce platforms typically show customers a delivery estimate generated from fixed rules, without accounting for seller location, product characteristics, shipping conditions, or a seller's actual delivery track record. Inaccurate estimates lead to customer dissatisfaction, complaints and refund requests, inefficient logistics planning, and erosion of customer trust when delivery promises are broken.

**Proposed solution:** A supervised machine learning regression system that predicts delivery time (in days) from historical order, product, seller, and customer data available at checkout.

```
Input:  Product category, seller location, freight value, order day, package weight, seller track record
Output: Estimated delivery time (days) + Promised SLA window

Example — Health & Beauty, seller in same state (SP), $18.5 freight, ordered on Monday
          → Estimated delivery time: 6.3 days (Promised Window: 4 – 9 days)
```

**Objectives:**
1. Develop an end-to-end machine learning pipeline to predict e-commerce delivery duration (`delivery_days`).
2. Analyze the geographic, operational, and commercial drivers of delivery delays.
3. Compare regression algorithms progressively from simple heuristics to state-of-the-art boosted ensembles.
4. Provide structured business insights and operational recommendations to optimize logistics planning.
5. Deploy a real-time interactive decision support web application for delivery estimation and risk diagnosis.

---

## Project Structure

```
DS Project/
├── app/
│   └── app.py                            # Interactive Streamlit delivery prediction web app
├── data/
│   ├── raw/                              # Original 9 Olist CSVs (orders, items, products, etc.)
│   └── processed/                        # Cleaned, merged, feature-engineered outputs
│       ├── item_level_clean.csv          # 110,178 item rows (36 columns)
│       └── order_level_clean.csv         # 96,460 aggregated order rows (20 columns)
├── models/
│   └── final_model.pkl                   # Exported best pipeline (Preprocessor + Tuned XGBoost)
├── notebooks/
│   ├── 01_data_understanding.ipynb       # Raw table inspection, types, missingness, skews
│   ├── 02_data_cleaning.ipynb            # Relational merge, integrity checks, leak-free feature engineering
│   ├── 03_eda.ipynb                      # Univariate/bivariate/correlation analysis & 9 business insights
│   ├── 04_model_baseline.ipynb           # Week 07: Problem framing, baseline & Linear Regression
│   └── 05_model_evaluation.ipynb         # Week 08: Multi-model comparison, 5-fold CV, tuning & error analysis
├── reports/
│   ├── data_quality_report.docx / .pdf   # Comprehensive data audit & cleaning documentation
│   ├── eda_summary.docx / .pdf           # Exploratory analysis & statistical findings
│   └── model_evaluation_summary.docx / .pdf # Model comparison, error diagnostics & business justification
├── README.md                             # Project overview and documentation
└── requirements.txt                      # Project dependency manifest
```

---

## Dataset

- **Source:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)
- **Size:** 99,441 raw orders across 5 relational tables (orders, order items, products, sellers, customers).
- **Time period:** October 2016 – August 2018.
- **Attributes used:** numerical (freight value, product weight, volume, basket size), categorical (product category, customer/seller state), temporal (purchase timestamp, day of week, weekend indicator), and relational location (seller state vs. customer state match).

---

## Complete Modeling & Evaluation Results

All candidate models were evaluated on the held-out test set (19,292 orders, 20%) under identical preprocessing (`StandardScaler` + `OneHotEncoder`):

| Model Architecture | Train MAE | 5-Fold CV MAE | Test MAE | Test RMSE | Test $R^2$ | Test MAPE | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Mean Baseline** | 6.412 d | $6.412 \pm 0.036$ | 6.433 d | 9.739 d | -0.000 | 81.9% | Benchmark |
| **Ridge Regression** | 5.103 d | $5.111 \pm 0.041$ | 5.101 d | 8.413 d | 0.254 | 52.9% | Baseline ML |
| **Random Forest (50 trees)** | 4.411 d | $4.767 \pm 0.032$ | 4.706 d | 8.153 d | 0.299 | 47.3% | Non-Linear |
| **XGBoost (Baseline)** | 4.604 d | $4.804 \pm 0.046$ | 4.754 d | 8.154 d | 0.299 | 47.9% | Gradient Boosted |
| **Tuned XGBoost (Final)** | **4.312 d** | $\mathbf{4.617 \pm 0.038}$ | $\mathbf{4.560\ d}$ | $\mathbf{8.002\ d}$ | $\mathbf{0.325}$ | $\mathbf{46.1\%}$ | **Production Selected** |

### Key Findings & Analytical Highlights
1. **Clear Superiority over Baseline:** Tuned XGBoost reduces average delivery prediction error from **6.43 days to 4.56 days** (a **29.1% relative improvement**), explaining nearly **32.5% of delivery duration variance**.
2. **Geography is the Strongest Signal:** Orders shipped within the customer's own state average **7.95 days** vs. **15.13 days** for cross-state orders — intra-state delivery cuts shipping duration in half ($r = -0.36$).
3. **Seller Dispatch History Dictates Latency:** Historical seller delivery duration ($r = +0.27$) is the single strongest positive delay driver.
4. **Weekend Invariance:** Weekend ordering has virtually zero correlation with delivery time ($r \approx 0.0039$), dispelling the myth that placing orders on weekends creates delivery backlogs.
5. **Segment Performance:** Same-state orders achieve a **3.05-day MAE** (Actual: 7.83d vs Predicted: 7.97d), while cross-state orders average a **5.42-day MAE**.

---

## Current Progress

- [x] Data collection (Raw Olist relational tables)
- [x] Data understanding (`01_data_understanding.ipynb`)
- [x] Data cleaning & relational merging (`02_data_cleaning.ipynb`)
- [x] Leak-free feature engineering (Leave-one-out seller history, geographic flags)
- [x] Exploratory data analysis & 9 structured business insights (`03_eda.ipynb`)
- [x] Week 07 Baseline Model deliverable (`04_model_baseline.ipynb`)
- [x] Week 08 Model Evaluation, CV, Tuning & Error Analysis (`05_model_evaluation.ipynb`)
- [x] Final model serialization (`models/final_model.pkl`)
- [x] Interactive Streamlit web deployment demo (`app/app.py`)
- [x] Academic reports in Word and PDF format (`reports/`)
- [x] Project dependencies manifest (`requirements.txt`)

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Jupyter Notebooks (in order)
```bash
# From the project root or notebooks/ directory:
jupyter notebook notebooks/01_data_understanding.ipynb
jupyter notebook notebooks/02_data_cleaning.ipynb
jupyter notebook notebooks/03_eda.ipynb
jupyter notebook notebooks/04_model_baseline.ipynb
jupyter notebook notebooks/05_model_evaluation.ipynb
```
*Every notebook is self-contained and runs top-to-bottom without errors.*

### 3. Launch the Interactive Streamlit Web App
```bash
streamlit run app/app.py
```
This opens the dynamic delivery estimation dashboard in your web browser, where you can select customer state, fulfillment geography, product category, package weight, and seller history to receive real-time estimates, promised SLA windows, and risk alerts.
