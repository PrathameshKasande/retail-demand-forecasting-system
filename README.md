<div align="center">📈 Retail Demand Forecasting System

Predict Future Sales · Optimize Inventory · Make Smarter Decisions

An end-to-end machine learning application that forecasts retail demand for the next 14 days and transforms historical sales data into actionable inventory replenishment recommendations.

""Live Demo" (https://img.shields.io/badge/🚀_Live_Demo-Open_Application-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)

""Python" (https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)" (https://www.python.org/)
""LightGBM" (https://img.shields.io/badge/LightGBM-Forecasting-2E7D32?style=flat-square)" (https://lightgbm.readthedocs.io/)
""Streamlit" (https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)" (https://streamlit.io/)
""Plotly" (https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=flat-square&logo=plotly&logoColor=white)" (https://plotly.com/)
""Docker" (https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)" (https://www.docker.com/)

"🚀 Live Demo" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/) · "📂 GitHub Repository" (https://github.com/PrathameshKasande/retail-demand-forecasting-system) · "🐛 Report a Bug" (https://github.com/PrathameshKasande/retail-demand-forecasting-system/issues) · "💡 Request a Feature" (https://github.com/PrathameshKasande/retail-demand-forecasting-system/issues)

</div>---

📑 Table of Contents

- "About the Project" (#-about-the-project)
- "Problem Statement" (#-problem-statement)
- "Key Features" (#-key-features)
- "Live Demo" (#-live-demo)
- "System Architecture" (#️-system-architecture)
- "Forecasting Methodology" (#-forecasting-methodology)
- "Model Performance" (#-model-performance)
- "Inventory Replenishment" (#-inventory-replenishment)
- "Technology Stack" (#️-technology-stack)
- "Project Structure" (#-project-structure)
- "Installation and Setup" (#-installation-and-setup)
- "Docker Deployment" (#-docker-deployment)
- "Input Data Guidelines" (#-input-data-guidelines)
- "Limitations" (#️-limitations)
- "Future Enhancements" (#-future-enhancements)
- "Contributing" (#-contributing)
- "Author" (#-author)

---

📌 About the Project

The Retail Demand Forecasting System is a time-series machine learning application designed to predict future product demand and support data-driven inventory planning.

Built with Python, LightGBM, and Streamlit, it processes historical sales data, engineers time-aware features, generates 14-day demand forecasts, and estimates inventory replenishment requirements.

The application supports business-level and product-level demand analysis, interactive visualizations, and downloadable forecast results.

🎯 Project Objectives

- Forecast future retail demand using historical sales patterns.
- Identify weekly seasonality and intermittent demand.
- Reduce temporal data leakage during model development.
- Visualize historical demand alongside future predictions.
- Estimate safety stock and replenishment quantities.
- Simplify inventory planning through an interactive dashboard.

🚨 Problem Statement

Retail businesses must maintain the right inventory levels to meet customer demand without tying up unnecessary capital.

Challenge| Business Impact
Overstocking| Higher storage costs and excess inventory
Stockouts| Lost sales and reduced customer satisfaction
Intermittent demand| Difficult-to-predict periods of zero sales
Seasonal fluctuations| Demand changes across weekdays and months
Bulk transactions| Outliers can distort normal demand patterns
Temporal data leakage| Misleading model evaluation and unreliable forecasts

💡 Proposed Solution

This project combines time-series feature engineering, LightGBM regression, chronological validation, interactive visualization, and inventory-planning calculations in one application.

It offers two analytical perspectives:

- Business-level forecasting: Analyze aggregated sales demand across the selected dataset.
- Product-level forecasting: Explore individual SKU demand and generate product-specific inventory recommendations.

---

✨ Key Features

<details>
<summary><strong>📊 14-Day Demand Forecasting</strong></summary>Uses a LightGBM regression model and historical demand features to predict sales over a 14-day forecast horizon.

</details><details>
<summary><strong>📂 Multi-Format Data Ingestion</strong></summary>Supports CSV, Excel, and PDF inputs through the application's configured document-processing pipeline.

</details><details>
<summary><strong>🧹 Data Cleaning and Preprocessing</strong></summary>Handles transaction cancellations, returns, date normalization, missing-date treatment, and configurable outlier filtering.

</details><details>
<summary><strong>📈 Interactive Visualizations</strong></summary>Uses Plotly charts to explore historical demand, predicted sales, and available uncertainty visualizations.

</details><details>
<summary><strong>📦 Inventory Replenishment Planning</strong></summary>Estimates future demand and safety stock to support purchasing decisions.

</details><details>
<summary><strong>⬇️ Downloadable Forecast Results</strong></summary>Exports available forecast and replenishment data to CSV for additional analysis.

</details><details>
<summary><strong>🐳 Docker Support</strong></summary>Provides a containerized deployment option for running the application in an isolated environment.

</details>---

🚀 Live Demo

<div align="center">Try the Application

Explore demand forecasts and inventory-planning features in the deployed application.

""Launch Application" (https://img.shields.io/badge/LAUNCH_LIVE_APPLICATION-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)

"Open Streamlit Application" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)

</div>Suggested usage:

1. Open the application.
2. Upload a supported sales dataset.
3. Review the processed demand data.
4. Explore the forecast visualization.
5. Review available inventory recommendations.
6. Export results when the application provides the download option.

«Tip: Add a screenshot of your actual dashboard here to give visitors an immediate preview of the application.»

<!-- Replace the path below with a real screenshot committed to your repository. --><!-- ![Retail Demand Forecasting Dashboard](assets/dashboard.png) -->---

🏗️ System Architecture

The application follows a data-to-decision pipeline, transforming raw transaction records into future demand forecasts and inventory-planning outputs.

flowchart TD
    A["📁 Sales Data Upload<br/>CSV / Excel / PDF"] --> B["🧹 Data Ingestion & Validation"]
    B --> C["🔄 Data Cleaning & Aggregation"]
    C --> D["📅 Daily Time-Series Construction"]
    D --> E["⚙️ Feature Engineering"]
    E --> F["🤖 LightGBM Regression Model"]
    F --> G["📈 14-Day Demand Forecast"]
    G --> H["📊 Interactive Dashboard"]
    G --> I["📦 Safety Stock Estimation"]
    I --> J["🛒 Replenishment Recommendations"]
    H --> K["⬇️ CSV Export"]
    J --> K

Pipeline Stages

Stage| Description
1. Data ingestion| Read supported sales documents
2. Data preprocessing| Clean transactions and prepare daily sales records
3. Feature engineering| Generate lag, rolling, and calendar features
4. Model inference| Generate demand predictions using LightGBM
5. Visualization| Compare historical sales with forecast results
6. Inventory planning| Estimate safety stock and replenishment needs
7. Export| Download available planning outputs

---

🧠 Forecasting Methodology

Forecast Horizon

The model is designed to forecast the next 14 days of retail demand.

Let (y_t) represent observed demand at time (t). The forecast horizon is:

[
H = 14
]

The feature pipeline uses historical observations available at the prediction origin to estimate future demand.

Feature Engineering

Feature Category| Examples| Purpose
Lag features| Sales at (t-14), (t-21), and (t-28)| Capture historical demand patterns
Rolling statistics| 7-day, 14-day, and 28-day means| Measure demand levels
Rolling volatility| Historical rolling standard deviation| Estimate demand variability
Intermittent demand| Frequency of zero-sales observations| Identify sparse demand patterns
Calendar features| Day of week, weekend, month, day of month| Capture calendar-related patterns
Target transformation| "log1p(y)" where configured| Reduce target skew

Temporal Data Leakage Prevention

Random train-test splitting can produce misleading results in time-series forecasting because observations from future dates may influence model training.

This project uses a chronological holdout approach with a reported 14-day validation horizon.

The intended safeguards include:

- Chronological training and validation boundaries.
- Historical features based on observations available at the forecast origin.
- Shifted lag and rolling features.
- Consistent feature construction between training and inference.

The actual leakage guarantee depends on the implementation of feature alignment, preprocessing, and validation logic.

LightGBM Model

The forecasting engine uses LightGBM, a gradient-boosted decision tree framework that learns nonlinear relationships between historical demand, calendar variables, and other engineered features.

Model configuration may include:

- Log-transformed demand targets.
- L2 regularization.
- Feature subsampling.
- Row subsampling.
- Chronological validation.

---

📊 Model Performance

The following scores are reported from the project's supplied chronological holdout evaluation.

Regression Metrics

Metric| LightGBM Model| Seasonal Baseline
RMSE| 51.59| 81.74
MAE| 24.12| 46.80
WMAPE| 63.83%| 93.46%

The reported results show lower errors than the selected seasonal baseline on the evaluated dataset.

Training vs. Validation

Metric| Reported Value
Training RMSE| 51.84
Validation RMSE| 51.59
Absolute difference| 0.25

A small difference between training and validation errors is useful to examine, but does not by itself prove the absence of overfitting. Both scores should be calculated on comparable target scales using a leakage-free pipeline.

Demand Activity Metrics

The project also reports these demand-activity classification metrics:

Metric| Reported Score
Accuracy| 98.93%
Precision| 99.12%
Recall| 99.62%
F1-score| 99.37%

These metrics should be accompanied by a clear description of the classification method and class definitions. If they are derived from regression predictions, document the thresholding procedure.

«Note: Reported performance is dataset-dependent and should be independently reproduced. Include the evaluation dates, dataset size, baseline definition, and metric calculation details when available.»

---

📦 Inventory Replenishment

The application translates demand forecasts into inventory-planning estimates.

1. Projected Demand

For a 14-day forecast:

[
D_{14} = \sum_{h=1}^{14}\hat{y}_{t+h}
]

where (\hat{y}_{t+h}) is the predicted demand for future day (h).

2. Safety Stock

A commonly used safety-stock approximation is:

[
SS = Z \times \sigma_d \times \sqrt{L}
]

Where:

- (SS): Safety stock.
- (Z): Service-level factor.
- (\sigma_d): Standard deviation of daily demand.
- (L): Lead time in days.

For a nominal 95% one-sided service level, (Z \approx 1.645), often rounded to 1.65.

The formula assumes an appropriate estimate of daily demand variability and suitable lead-time and demand-dependence assumptions.

3. Replenishment Quantity

A basic inventory-planning formula is:

[
Q = \max(0, D_L + SS - I)
]

Where:

- (Q): Recommended order quantity.
- (D_L): Expected demand over lead time.
- (SS): Safety stock.
- (I): Inventory position, including relevant on-hand inventory and outstanding orders.

Actual purchase-order calculations depend on the inventory inputs and calculation logic implemented in the application.

Available Planning Outputs

- Projected demand.
- Daily forecast values.
- Safety-stock estimates.
- Replenishment recommendations, where supported by available inputs.
- Downloadable CSV results.

---

🛠️ Technology Stack

Layer| Technologies
Programming language| Python 3.11
Machine learning| LightGBM, scikit-learn
Data processing| Pandas, NumPy, PyArrow
Web application| Streamlit
Data visualization| Plotly, Matplotlib, Seaborn
PDF processing| pdfplumber
Excel processing| openpyxl
Containerization| Docker
Cloud deployment| Streamlit Community Cloud

---

📂 Project Structure

retail-demand-forecasting-system/
│
├── .streamlit/
│   └── config.toml
│
├── config/
│   └── config.yaml
│
├── data/
│   └── sample/
│       └── sample_sales.csv
│
├── models/
│   └── lightgbm_forecast_model.txt
│
├── notebooks/
│   └── 01_eda_and_baseline.ipynb
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── app.py
└── requirements.txt

Important Files

- "app.py" — Main Streamlit application.
- "requirements.txt" — Python dependencies.
- "config/config.yaml" — Forecasting and model configuration.
- "models/lightgbm_forecast_model.txt" — Serialized model, if included in the repository.
- "data/sample/sample_sales.csv" — Sample sales dataset.
- "notebooks/01_eda_and_baseline.ipynb" — Exploratory analysis and model development.
- "Dockerfile" — Container build configuration.
- ".streamlit/config.toml" — Streamlit interface configuration.

---

💻 Installation and Setup

Prerequisites

- Python 3.11
- Git
- pip
- Docker (optional)

Step 1: Clone the Repository

git clone https://github.com/PrathameshKasande/retail-demand-forecasting-system.git
cd retail-demand-forecasting-system

Step 2: Create a Virtual Environment

Windows — PowerShell

python -m venv venv
.\venv\Scripts\Activate.ps1

macOS / Linux

python3.11 -m venv venv
source venv/bin/activate

Step 3: Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

Step 4: Run the Application

streamlit run app.py

Open the local URL printed in the terminal, usually:

"http://localhost:8501" (http://localhost:8501)

---

🐳 Docker Deployment

Run the application inside a Docker container.

Build the Image

docker build -t retail-demand-forecasting .

Start the Container

docker run --rm -p 8501:8501 retail-demand-forecasting

Open "http://localhost:8501" (http://localhost:8501).

Ensure that the Docker configuration starts Streamlit on port "8501" and binds the server to "0.0.0.0".

---

📄 Input Data Guidelines

Depending on the selected input format and the application's parser, the following fields may be useful:

Field| Description
Transaction date| Date of the sale
Product identifier| SKU or product code
Quantity| Number of units sold or returned
Invoice identifier| Transaction reference
Unit price| Optional transaction value

Data Preparation Recommendations

1. Use consistent date formats.
2. Validate numeric quantity values.
3. Handle returns and cancellations consistently.
4. Distinguish true zero-demand days from missing records.
5. Review unusually large transactions before removing them.
6. Provide sufficient historical data to capture demand patterns.

Accepted column names and file schemas depend on the implementation. Refer to the sample dataset and application code for the expected input format.

---

⚠️ Limitations

- Forecast quality depends on the completeness and reliability of historical data.
- Unexpected promotions, holidays, and supply disruptions can affect demand.
- Recorded zero sales may reflect unavailable inventory rather than zero customer demand.
- Filtering genuine bulk orders may understate wholesale demand.
- Safety-stock estimates depend on demand variability and lead-time assumptions.
- Forecast accuracy metrics do not guarantee future business outcomes.
- Review data privacy requirements before uploading confidential business information to a hosted application.

---

🔮 Future Enhancements

- [ ] Automated hyperparameter tuning.
- [ ] Rolling-origin time-series backtesting.
- [ ] Prediction intervals calibrated using forecast errors.
- [ ] Product-specific lead-time integration.
- [ ] Promotion and holiday-aware forecasting.
- [ ] Automated model retraining and drift monitoring.
- [ ] Unit testing and continuous integration.
- [ ] SHAP-based model explainability.
- [ ] Multi-store and multi-warehouse forecasting.
- [ ] Inventory-aware forecast accuracy monitoring.

---

🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Implement and test your changes.
4. Submit a pull request describing your improvements.
.

Use the "GitHub Issues page" (https://github.com/PrathameshKasande/retail-demand-forecasting-system/issues) to report bugs or request features.

---

📜 License

A license has not yet been specified in this README. Add a "LICENSE" file and update this section before distributing the project for reuse.

---

👨‍💻 Author

<div align="center">Prathamesh Kasande

Bachelor of Technology in Computer Science and Engineering

""GitHub" (https://img.shields.io/badge/GitHub-PrathameshKasande-181717?style=for-the-badge&logo=github)" (https://github.com/PrathameshKasande)

"View Repository" (https://github.com/PrathameshKasande/retail-demand-forecasting-system) · "Explore Live Demo" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)

</div>---

<div align="center">⭐ If you find this project interesting, consider starring the repository!

Built with Python, LightGBM, and Streamlit.

Turning historical retail sales into actionable demand forecasts.

</div>
