# 📦 Retail Demand Forecasting & Inventory Optimization Engine

An end-to-end machine learning system forecasting multi-SKU retail unit demand across a 14-day horizon. Engineered to eliminate temporal lookahead data leakage, benchmarked using asymmetric business inventory losses and supply-chain metrics (WMAPE, RMSE, F1-Score), and containerized for deployment.

---

## 🚀 Key Features

* **Zero Lookahead Data Leakage:** Implemented shifted lag windows ($t-14, t-21, t-28$) and rolling statistics strictly offset by the forecast horizon.
* **Dual Evaluation Suite:**
  * **Continuous Regression:** Evaluated on RMSE, MSE, MAE, and WMAPE.
  * **Discrete Event Classification:** Assesses ability to distinguish active demand days from zero-sales days (Accuracy, Precision, Recall, F1 Score).
* **Asymmetric Business Loss:** Optimizes for supply chain realities—penalizing out-of-stock lost revenue ($4.0/unit) more heavily than holding costs ($1.0/unit).
* **Single-File Streamlit UI:** Interactive dashboard for supply planners to review forecasts, prediction intervals, and automated safety stock calculations.
* **CI/CD Automation:** GitHub Actions pipeline running Flake8 linting, zero-leakage unit tests, and Docker container build validation.

---

## ⚡ Quick Start

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run unit tests
python -m pytest tests/

# 3. Train model and evaluate
python -m src.train

# 4. Launch web app
python -m streamlit run app.py