<div align="center">📈 Retail Demand Forecasting System

Predict Future Sales. Optimize Inventory. Make Smarter Decisions.

A machine learning application that forecasts retail demand for the next 14 days and helps businesses make better inventory replenishment decisions.

""Live Demo" (https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge)" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)
""Python" (https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)" (https://www.python.org/)
""Streamlit" (https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)" (https://streamlit.io/)
""LightGBM" (https://img.shields.io/badge/LightGBM-Machine_Learning-2E7D32?style=for-the-badge)" (https://lightgbm.readthedocs.io/)

"🚀 Try Live Demo" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/) • "🐛 Report Bug" (https://github.com/PrathameshKasande/retail-demand-forecasting-system/issues) • "💡 Request Feature" (https://github.com/PrathameshKasande/retail-demand-forecasting-system/issues)

</div>---

📌 About the Project

The Retail Demand Forecasting System uses machine learning to predict future retail sales from historical transaction data.

It helps users understand demand patterns, visualize future sales, estimate safety stock, and plan inventory replenishment.

✨ Key Features

- 📊 14-Day Sales Forecasting — Predict future demand using LightGBM.
- 📁 Multiple File Formats — Support for CSV, Excel, and PDF inputs, subject to the application's accepted formats.
- 🧹 Data Cleaning — Handle cancellations, missing dates, and unusual sales transactions.
- 📈 Interactive Charts — Explore historical sales and future forecasts.
- 📦 Inventory Planning — Estimate safety stock and replenishment quantities.
- ⬇️ CSV Export — Download forecast results for further analysis.
- 🐳 Docker Support — Run the application in a container.

🚀 Live Demo

👉 "Click here to launch the application" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)

Upload your sales data, explore the forecast, and review the available inventory recommendations.

🛠️ Tech Stack

Technology| Purpose
Python| Core programming language
Pandas & NumPy| Data processing
LightGBM| Demand forecasting
Scikit-learn| Machine learning utilities and evaluation
Streamlit| Interactive web application
Plotly| Interactive charts
pdfplumber & openpyxl| Document processing
Docker| Containerization

⚙️ How It Works

flowchart TD
    A["📁 Upload Sales Data"] --> B["🧹 Data Cleaning"]
    B --> C["⚙️ Feature Engineering"]
    C --> D["🤖 LightGBM Model"]
    D --> E["📈 14-Day Forecast"]
    E --> F["📦 Inventory Recommendations"]
    F --> G["⬇️ Download Results"]

1. Upload historical sales data.
2. Clean and prepare the dataset.
3. Generate historical demand and calendar features.
4. Predict demand for the next 14 days.
5. Visualize forecast results and review inventory recommendations.
6. Download the available results as a CSV file.

📂 Project Structure

retail-demand-forecasting-system/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
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
└── .streamlit/
    └── config.toml

💻 Installation & Setup

1. Clone the Repository

git clone https://github.com/PrathameshKasande/retail-demand-forecasting-system.git
cd retail-demand-forecasting-system

2. Create a Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate

macOS / Linux:

python3.11 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Run the Application

streamlit run app.py

Open the local URL displayed in your terminal, usually "http://localhost:8501".

🐳 Run with Docker

docker build -t retail-demand-forecasting .
docker run --rm -p 8501:8501 retail-demand-forecasting

Open "http://localhost:8501" in your browser.

📊 Model Evaluation

The project reports the following results from its 14-day chronological holdout evaluation.

Metric| Model Score
RMSE| 51.59
MAE| 24.12
WMAPE| 63.83%

Results are dataset-dependent and should be reproduced against the documented validation period before being treated as production performance guarantees.

🔮 Future Improvements

- [ ] Automated model retraining
- [ ] More advanced forecasting models
- [ ] Improved prediction intervals
- [ ] Inventory and lead-time integration
- [ ] Automated testing and continuous integration

👨‍💻 Author
Prathamesh Kasande
Bachelor of Technology in Computer Science and Engineering

- "GitHub Profile" (https://github.com/PrathameshKasande)
- "Project Repository" (https://github.com/PrathameshKasande/retail-demand-forecasting-system)
- "Live Application" (https://retail-demand-forecasting-system-hogf6rwssgakvdibbc58oe.streamlit.app/)

---

<div align="center">⭐ If you find this project interesting, consider giving the repository a star!

Built with Python, LightGBM, and Streamlit.

</div>

