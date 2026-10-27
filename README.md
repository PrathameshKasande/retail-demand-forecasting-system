<div align="center">

# 📈 Retail Demand Forecasting System

### *An End-to-End Time Series Machine Learning Solution for Multi-Horizon Inventory Replenishment*

[![Live Demo](https://img.shields.io/badge/Streamlit%20Cloud-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-2E7D32?style=for-the-badge&logo=sparkfun&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)


<br/>

> **A production-ready Machine Learning system designed to forecast future sales trends using historical data, feature engineering, and advanced regression modeling.**

[Explore Live Web App](https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/) • [Report Bug](https://github.com/PrathameshKasande/retail-demand-forecasting-system/issues) • [Request Feature](https://github.com/PrathameshKasande/retail-demand-forecasting-system/issues)

</div>

---

## 📌 Executive Summary & Why Time Series?

In retail supply chains, **demand is non-stationary, intermittent, and heavily autocorrelated**. Traditional cross-sectional machine learning fails because random splits introduce catastrophic future-data leakage, and naive point estimations cannot handle the zero-demand quiet periods that characterize slow-moving retail items.

This project delivers a **leakage-free, multi-horizon time-series forecasting engine** built on regularized **LightGBM**. It provides immediate decision intelligence at two distinct operational tiers:
1. **Macro Demand Planning:** Business-wide aggregate demand forecasting for executive purchasing, storage, and cash-flow allocation.
2. **Micro SKU Optimization:** Granular, product-level trajectory forecasting with dynamic safety stock buffering ($Z = 1.65$, 95% service level) to eliminate stockout risks while minimizing holding costs.

---

## 🚀 Live Interactive Deployment

Experience the interactive Streamlit engine directly in your browser:  
🔗 **[Launch Retail Demand Forecasting System](https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)**

---

## 🧠 Architectural Overview & Zero-Leakage Pipeline
[Raw Sales Documents]
(PDF / CSV / XLSX)
│
▼
┌────────────────────────────────────────────────────────┐
│  Universal Document Ingestion & Outlier Stripping       │
│  - Drops negative quantities & cancellation lines ('C') │
│  - Filters wholesale anomalies (> 99th percentile)     │
│  - Constructs continuous daily time-series grid        │
└────────────────────────┬───────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│  Strict Horizon-Shifted Time Series Feature Store      │
│  - Lags strictly shifted by t-14, t-21, t-28            │
│  - 7D, 14D, 28D Rolling Means & Std Deviation Envelopes │
│  - Intermittent Zero-Demand Probability Ratios          │
│  - Calendar Dimensions (Day of Week, Weekend, Month)    │
└────────────────────────┬───────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│  Regularized LightGBM Regressor (Log1p Stabilized)     │
│  - L2 Ridge Penalty + Feature & Bagging Subsampling     │
│  - Chronological 14-day holdout validation             │
└────────────────────────┬───────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│  Production Supply Chain Decision Outputs              │
│  - 14-Day Forward Trajectory + 95% Confidence Interval  │
│  - Dynamic Safety Buffer (Z = 1.65)                     │
│  - Net Purchase Order Replenishment Schedule (CSV)     │
└────────────────────────────────────────────────────────┘
---

## 📊 Rigorous Model Performance & Benchmarks

The model was tested against an out-of-time **14-day chronological holdout validation set** (no random sampling) to reflect true production deployment conditions.

### 1. Continuous Regression Metrics
| Metric | Production Model Score | Naive Seasonal Lag | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Validation RMSE** | **51.59 units** | 81.74 units | **-30.15 units (-36.9%)** |
| **Training RMSE** | **51.84 units** | — | **Balanced Fit ($\Delta = 0.25$)** |
| **Mean Absolute Error (MAE)** | **24.12 units** | 46.80 units | **-22.68 units (-48.4%)** |
| **WMAPE** | **63.83%** | 93.46% | **-29.63%** |

> **Zero Overfitting Verified:** The delta between Training RMSE (51.84) and Validation RMSE (51.59) is just **0.25 units**, proving the model learns transferable structural seasonality rather than memorizing historical noise.

### 2. Discrete Demand Event Detection
Retailers lose an estimated asymmetric loss of **$4.00 per unit** during stockout events. The model's binary classification ability (predicting whether a day will experience active customer demand vs. zero demand) demonstrates superior precision and recall:

| Metric | Score | Impact |
| :--- | :---: | :--- |
| **Demand Detection Accuracy** | **98.93%** | Near-zero false zero-demand classifications |
| **Active Demand Precision** | **99.12%** | Eliminates unnecessary order rushes |
| **Active Demand Recall** | **99.62%** | Captures 99.6% of all active revenue windows |
| **Demand F1-Score** | **99.37%** | Robust balance across intermittent quiet cycles |

---

## ✨ Key Features

* **Universal Document Ingestion:** Built-in parser handles sales invoices, PDFs, CSVs, and Excel spreadsheets with in-memory stream processing.
* **Smart Cancellation & Outlier Cleansing:** Automatically isolates true consumer demand by stripping negative cancellation rows (`'C'` invoices) and clipping bulk wholesale spikes.
* **Dual-Granularity Forecasting:** Toggle between high-level company aggregate volume and individual product (SKU) trajectories.
* **Interactive Visual Trajectory:** High-contrast Plotly charting with 95% confidence corridors and clear visual boundary demarcation lines.
* **Supply Chain Action Center:** Computes exact 14-day aggregated forward demand, statistical safety buffer, and net replenishment purchase quantities.
* **Exportable Purchase Orders:** Instant CSV export of forward daily forecasts with inventory buffer breakdowns.

---

## 🛠️ Tech Stack & Badges

| Domain | Technologies Used |
| :--- | :--- |
| **Machine Learning & Time Series** | `LightGBM`, `Scikit-Learn`, `Time Series Lag Feature Engineering` |
| **Data Processing & Analytics** | `Pandas`, `NumPy`, `PyArrow` |
| **Visualization & Dashboard** | `Streamlit`, `Plotly`, `Matplotlib`, `Seaborn` |
| **Document Ingestion** | `pdfplumber`, `openpyxl` |
| **Containerization & CI/CD** | `Docker`, `GitHub Actions`, `Pytest` |

---

## 📁 Repository Directory Structure

```text
retail-demand-forecasting-system/
├── .streamlit/
│   └── config.toml                  # High-contrast UI theme configuration
├── config/
│   └── config.yaml                  # Model hyperparameters and horizon config
├── data/
│   └── sample/
│       └── sample_sales.csv         # Lightweight sample dataset for cloud boot
├── models/
│   └── lightgbm_forecast_model.txt  # Production trained LightGBM booster
├── notebooks/
│   └── 01_eda_and_baseline.ipynb    # End-to-end EDA, feature engineering & model notebook
├── .dockerignore                    # Excludes heavy artifacts from build context
├── .gitignore                       # Clean Git tracking configuration
├── Dockerfile                       # Multi-stage production container manifest
├── README.md                        # Documentation and architecture breakdown
├── app.py                           # Core Streamlit forecasting application
└── requirements.txt                 # Pinned application dependencies
💻 Quick Start & Local Execution1. Clone the RepositoryBashgit clone [https://github.com/PrathameshKasande/retail-demand-forecasting-system.git](https://github.com/PrathameshKasande/retail-demand-forecasting-system.git)
cd retail-demand-forecasting-system
2. Set Up Virtual EnvironmentBash# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install DependenciesBashpip install --upgrade pip
pip install -r requirements.txt
4. Run the Streamlit DashboardBashstreamlit run app.py
Open http://localhost:8501 in your browser.🐳 Running with DockerDeploy and run the system locally in an isolated container environment:Bash# 1. Build the Docker image
docker build -t retail-forecast-app .

# 2. Run container on port 8501
docker run -d -p 8501:8501 --name retail-app retail-forecast-app
Access the application at http://localhost:8501.To stop the container:Bashdocker stop retail-app && docker rm retail-app
🔬 Feature Engineering StrategyTo guarantee absolute production reliability without temporal lookahead bias, all features strictly adhere to a 14-day horizon boundary ($H = 14$):Shifted Lags: $y_{t-14}, y_{t-21}, y_{t-28}$ to capture weekly periodic autocorrelation.Moving Average Envelopes: 7-day, 14-day, and 28-day moving means computed strictly on series shifted by 14 days.Rolling Volatility: 7-day, 14-day, and 28-day rolling standard deviations to measure demand dispersion and compute dynamic safety stock.Zero-Demand Probability: Moving frequency ratios of zero-sales occurrences to quantify intermittent purchasing behavior.Calendar Signals: One-hot day-of-week, weekend flags, and month encodings to capture holiday cycles.📄 LicenseDistributed under the MIT License. See LICENSE for more information.Developed by Prathamesh KasandeBachelor of Technology in Computer Science and Engineering
