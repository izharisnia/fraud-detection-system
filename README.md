# Operational Fraud Detection System

An enterprise-grade, real-time machine learning system designed to detect, analyze, and prevent financial transaction fraud across high-volume banking and payment networks. 

Built with **LightGBM**, **Scikit-Learn**, **Streamlit**, and **Plotly**, this system processes over 6.36 million financial records to flag high-risk transactions, uncover illicit money-laundering hubs, and minimize financial exposure while maintaining low customer friction.

---

## Technical Performance Benchmarks

| Metric | Score / Benchmark | Description |
| :--- | :--- | :--- |
| **ROC-AUC Score** | **0.9914** | Area under ROC Curve evaluating ranking capability across all thresholds |
| **Recall Catch Rate** | **99.14%** | Percentage of total financial fraud cases correctly intercepted |
| **F2 Score** | **0.9840** | Cost-tuned metric prioritizing Recall over Precision |
| **Optimal Threshold** | **0.10** | Balanced decision boundary minimizing operational false alarms |
| **Net Capital Saved** | **$3,175,350.00** | Net financial loss prevented across Kaggle validation test split |
| **Inference Latency** | **~12 ms / req** | Real-time payload evaluation latency |

---

## Key System Features

1. **Operational Analytics Dashboard**:
   * Dual-axis monthly fraud trend monitoring against total transaction volume.
   * Dynamic feature importance breakdown and model explainability.
   * Real-time profile metrics derived directly from `Fraud.csv` (6.35M+ unique profiles).

2. **Real-Time Transaction Risk Engine**:
   * Interactive single-payload risk scoring engine.
   * Returns calibrated fraud probability scores and automated operational risk factor alerts (e.g., account liquidations, off-peak night transfers).

3. **Batch Scanner & Export**:
   * Multi-row CSV batch ingestion pipeline.
   * Automated risk scoring, risk ratio calculation, capital at risk quantification, and scored CSV export.

4. **Network Topology Graph**:
   * Directed node graph built with NetworkX and Plotly visualizing transaction flows.
   * Pinpoints suspicious recipient hubs acting as money mule collectors.

5. **Integrated Interview Masterclass (Tab 5)**:
   * Embedded master guide covering end-to-end architecture, feature engineering formulas, model selection rationale, business cost-benefit matrices, and top 15 technical interview questions.

---

## Project Structure

```
Fraud_Detection/
├── app.py                      # Main Streamlit web application & UI engine
├── preprocessing.py            # Feature engineering, scaling & preprocessing pipeline
├── train_and_save.py           # LightGBM model training, threshold tuning & artifact generation
├── interview_guide.md          # 661-line masterclass documentation & interview preparation guide
├── requirements.txt            # Python package dependencies
├── Dockerfile                  # Containerized deployment specification
├── metadata.json               # Trained model metrics, threshold & feature metadata
├── fraud_model.pkl             # Trained LightGBM model artifact
├── scaler.pkl                  # Fitted StandardScaler artifact
└── .streamlit/
    └── config.toml             # Streamlit server & UI theme configurations
```

---

## Quick Start Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Installation
Clone the repository and install requirements:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt
```

### 3. Model Training & Artifact Generation
To retrain the model or regenerate pipeline artifacts:

```bash
python train_and_save.py
```

### 4. Launch Local Web Application
Run the Streamlit application:

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Production Deployment Options

### 1. Streamlit Community Cloud (100% Free)
1. Push this repository to **GitHub**.
2. Connect your repository at **[share.streamlit.io](https://share.streamlit.io)**.
3. Set main file path to `app.py` and click **Deploy**.

### 2. Docker Container Deployment
Build and run using Docker:

```bash
docker build -t fraud-detection-app .
docker run -p 8501:8501 fraud-detection-app
```

---

## Dataset Information

* **Source**: Kaggle Financial Fraud Dataset (`Fraud.csv`)
* **Total Transactions**: 6,362,620 rows (493.5 MB)
* **Key Columns**: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`.

---

## License & Compliance

Designed in accordance with **ISO 27001** & **PCI-DSS** data security standards. Free for operational, academic, and interview preparation use.
