<div align="center">📈 Retail Demand Forecasting System

Intelligent Demand Prediction & Inventory Optimization

Forecast smarter. Stock better. Reduce inventory risk.

A machine learning application that predicts retail demand for the next 14 days and transforms historical sales data into actionable inventory replenishment insights.

<br/><a href="https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀_EXPLORE_LIVE_DEMO-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/>
</a>
<a href="https://github.com/PrathameshKasande/retail-demand-forecasting-system">
  <img src="https://img.shields.io/badge/VIEW_SOURCE_CODE-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code"/>
</a><br/><br/>

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/LightGBM-ML_Model-2E7D32?style=flat-square" alt="LightGBM"/>
<img src="https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
<img src="https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly"/>
<img src="https://img.shields.io/badge/Docker-Supported-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"/></div>---

🎯 Overview

Retail businesses need to balance product availability with inventory costs. This project uses LightGBM-based time-series forecasting to estimate future sales and support better purchasing decisions.

The application combines data preprocessing, historical demand feature engineering, machine learning, interactive visualizations, and inventory replenishment calculations in a single Streamlit dashboard.

✨ Key Highlights

- 📊 14-Day Forecasting — Predict upcoming retail demand.
- 🤖 LightGBM Regression — Model nonlinear demand patterns.
- 🛡️ Leakage-Aware Features — Use historical information aligned with the forecasting horizon.
- 📦 Inventory Optimization — Estimate safety stock and replenishment requirements.
- 📈 Interactive Dashboard — Explore historical sales and forecast trajectories.
- 📁 Multiple Input Formats — Process supported CSV, Excel, and PDF files.
- ⬇️ CSV Export — Download available forecast and planning results.
- 🐳 Docker Support — Run the application in a containerized environment.

🚀 Live Application

<div align="center">See the Forecasting System in Action

"Open the Live Dashboard →" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)

</div>The application is designed to let users upload sales data, explore demand forecasts, review inventory recommendations, and export results.

<!-- Add a screenshot of your actual application here after committing it to the repository. --><!-- ![Dashboard Preview](assets/dashboard.png) -->---

⚙️ How It Works

flowchart TD
    A["📁 Sales Data"] --> B["🧹 Data Preprocessing"]
    B --> C["⚙️ Feature Engineering"]
    C --> D["🤖 LightGBM Model"]
    D --> E["📈 14-Day Forecast"]
    E --> F["📦 Safety Stock & Replenishment"]
    F --> G["📊 Interactive Dashboard"]
    G --> H["⬇️ CSV Export"]

    style A fill:#E0F2FE,stroke:#0284C7,color:#0F172A
    style D fill:#DCFCE7,stroke:#16A34A,color:#14532D
    style E fill:#EDE9FE,stroke:#7C3AED,color:#312E81
    style G fill:#FFE4E6,stroke:#E11D48,color:#881337

Forecasting Methodology

Component| Description
Historical lags| Capture previous demand patterns
Rolling statistics| Estimate demand level and volatility
Calendar features| Capture weekday and seasonal patterns
Target transformation| Apply "log1p(y)" where configured
Validation| Evaluate using a chronological holdout
Forecast horizon| Next 14 days

The feature pipeline is designed to avoid temporal leakage by using historical information available at the forecast origin. Correctness depends on feature alignment and the implementation of the training and inference pipeline.

---

📊 Model Performance

Reported results from the project's 14-day chronological holdout evaluation:

Metric| LightGBM Model| Seasonal Baseline
RMSE| 51.59| 81.74
MAE| 24.12| 46.80
WMAPE| 63.83%| 93.46%

The reported results show lower errors than the selected seasonal baseline on the evaluated dataset.

<details>
<summary><strong>📈 View Additional Evaluation Metrics</strong></summary>Metric| Reported Result
Training RMSE| 51.84
Validation RMSE| 51.59
Demand activity accuracy| 98.93%
Active demand precision| 99.12%
Active demand recall| 99.62%
Demand activity F1-score| 99.37%

These results should be verified against the actual evaluation code. In particular, document how demand activity is classified and how the training and validation metrics are calculated.

</details>«Note: Model performance depends on the dataset, validation period, and evaluation methodology. Reported results are not guarantees of future forecasting accuracy.»

---

📦 Inventory Replenishment

The application uses forecast demand and statistical inventory calculations to support replenishment planning.

Safety Stock

[
SS = Z \times \sigma_d \times \sqrt{L}
]

Where:

- (SS) = Safety stock
- (Z) = Service-level factor
- (\sigma_d) = Standard deviation of daily demand
- (L) = Replenishment lead time in days

For a nominal 95% one-sided service level, (Z) is approximately 1.645.

Basic Replenishment Estimate

[
Q = \max(0, D_L + SS - I)
]

Where (D_L) is expected lead-time demand and (I) is the relevant inventory position.

The accuracy of these recommendations depends on the inventory inputs, lead-time assumptions, and calculations implemented in the application.

---

🛠️ Tech Stack

Technology| Purpose
Python 3.11| Application development
Pandas & NumPy| Data processing
LightGBM| Demand forecasting
Scikit-learn| Machine learning utilities
Streamlit| Interactive web interface
Plotly| Data visualization
pdfplumber & openpyxl| Document processing
Docker| Containerization

---

📂 Project Structure

retail-demand-forecasting-system/
│
├── .streamlit/
│   └── config.toml
├── config/
│   └── config.yaml
├── data/
│   └── sample/
│       └── sample_sales.csv
├── models/
│   └── lightgbm_forecast_model.txt
├── notebooks/
│   └── 01_eda_and_baseline.ipynb
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md

---

💻 Installation & Setup

1. Clone the Repository

git clone https://github.com/PrathameshKasande/retail-demand-forecasting-system.git
cd retail-demand-forecasting-system

2. Create a Virtual Environment

Windows — PowerShell

python -m venv venv
.\venv\Scripts\Activate.ps1

macOS / Linux

python3.11 -m venv venv
source venv/bin/activate

3. Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Launch the Application

streamlit run app.py

Open "http://localhost:8501" (http://localhost:8501) in your browser.

---

🐳 Run with Docker

docker build -t retail-demand-forecasting .
docker run --rm -p 8501:8501 retail-demand-forecasting

Then visit "http://localhost:8501" (http://localhost:8501).

Ensure the Docker configuration starts Streamlit on port "8501" and binds the server to "0.0.0.0".

---

🔮 Future Improvements

- [ ] Automated model retraining and monitoring
- [ ] Rolling-origin time-series backtesting
- [ ] Improved prediction intervals
- [ ] Product-specific lead-time integration
- [ ] Promotion and holiday-aware forecasting
- [ ] SHAP-based model explainability
- [ ] Automated testing and continuous integration
- [ ] Multi-store and multi-warehouse forecasting

🤝 Contributing

Contributions and suggestions are welcome.

1. Fork this repository.
2. Create a feature branch.
3. Make and test your changes.
4. Submit a pull request.

"Report a bug or request a feature →" (https://github.com/PrathameshKasande/retail-demand-forecasting-system/issues)

👨‍💻 Author

<div align="center">Prathamesh Kasande

Bachelor of Technology in Computer Science and Engineering

""GitHub" (https://img.shields.io/badge/GitHub-PrathameshKasande-181717?style=for-the-badge&logo=github)" (https://github.com/PrathameshKasande)

"Project Repository" (https://github.com/PrathameshKasande/retail-demand-forecasting-system) · "Live Demo" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)

</div>---

<div align="center">⭐ If you find this project interesting, consider giving it a star!

Turning historical retail sales into actionable demand forecasts.

</div>
