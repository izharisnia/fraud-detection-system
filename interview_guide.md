# Project Description of Fraud Detection System

Great question — I will explain your entire fraud detection project from start to end in clear, executive language, step by step. You can use this explanation for learning, technical interviews, or corporate presentations.

---

## Technical Overview – Fraud Detection System

### 1. What Problem Are We Solving?
Banks process millions of transactions every day. Some of them are fraudulent (unauthorized fund transfers).  
**Goal**: Automatically identify fraudulent transactions using Machine Learning in real time before funds leave the banking network.

### 2. What Data Are We Using?
- Kaggle Financial Fraud Dataset (`Fraud.csv`)
- Around 6.36 million transactions (493.5 MB)
- Transaction Attributes:
  - Sender Account (`nameOrig`)
  - Receiver Account (`nameDest`)
  - Transaction Amount (`amount`)
  - Time Simulation Step (`step`: 1 to 744 hours = 30 days)
  - Account Balances (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`)
  - Target Label (`isFraud`: 1 = fraud, 0 = normal)

### 3. Why Machine Learning Is Needed
Fraud is:
- Extremely Rare (~0.13% of all transactions)
- Hidden inside normal transaction traffic
- Dynamic and evolving over time

Static rules like `amount > $200,000 = fraud` fail because fraudsters split transactions.  
Machine Learning:
- Learns non-linear behavioral patterns from historical data
- Uncovers hidden complex relationships across features
- Adapts dynamically to emerging fraud vectors

### 4. Step 1: Data Preprocessing (Cleaning & Memory Optimization)
What we do here:
- Load data efficiently using custom 32-bit/8-bit `dtypes`
- Reduce memory usage by 70% (preventing system RAM crashes)
- Remove non-predictive identifiers before model training

Why:
- Large datasets (6.36M rows) require memory optimization
- High-quality clean features improve model accuracy

### 5. Step 2: Feature Engineering (Core System Intelligence)
Raw data alone is insufficient. We engineer domain-specific financial features:

#### a) Balance Consistency Checks
We check:
- Does money subtraction/addition match account balances?
- Fraudulent transactions frequently break balance calculation logic.

#### b) Transaction Impact Features
We calculate:
- Ratio of transaction amount relative to sender initial balance.
- Fraudsters typically attempt to transfer a large fraction of available funds.

#### c) Time-Based Features
We extract:
- Hour of the day from cumulative step counter
- Binary flag indicating off-peak night execution (12 AM – 6 AM)
- Fraud probability spikes during off-peak hours when account owners sleep.

#### d) Sender Behavioral Features
For each sender account:
- Cumulative transaction count
- Historical average transaction amount
- Sudden deviations from historical baselines indicate account takeover.

#### e) Receiver Behavioral Features
For each recipient account:
- Incoming transaction frequency
- Total incoming volume
- Mule collector accounts receive transfers from multiple unique senders.

#### f) Network Topology Features
We analyze:
- Recipient graph hub connectivity
- Historical fraud rates of incoming connections
- Exposes organized money laundering networks.

### 6. Step 3: Training Data Preparation
- **X**: Input feature matrix (transaction behavior signals)
- **y**: Target binary vector (`isFraud`)
- Account IDs (`nameOrig`, `nameDest`) are removed prior to training to eliminate Data Leakage and force the model to learn behavioral patterns rather than memorizing account strings.

### 7. Step 4: Model Selection & Architecture
- Primary Classifier: **LightGBM Gradient Boosting Model**
- Secondary Fallback: **Scikit-Learn HistGradientBoostingClassifier** (automatically engaged if C++ `libomp` runtimes are absent on macOS)

Why LightGBM?
- Optimized for large-scale tabular datasets (6.36M rows)
- Efficient handling of sparse categorical and imbalanced data
- Fast leaf-wise tree growth with lower memory consumption

### 8. Step 5: Model Training & Validation Split
- Split: **70% Training / 30% Stratified Validation**
- Stratified sampling ensures equal representation of the 0.13% minority fraud class in both train and test splits.

### 9. Step 6: Probability Scoring
The model outputs a continuous calibrated risk score ($p \in [0.0, 1.0]$):
- `0.02` -> Low Risk (Automated Approval)
- `0.85` -> High Risk (Automated Block & Fraud Alert)

### 10. Step 7: Decision Threshold Tuning
Default threshold (0.50) misses ~40% of subtle fraud cases.  
We apply cost-matrix economic optimization to set an optimal threshold of **0.10**:
- Catch Rate (Recall): **99.14%**
- Net Financial Capital Saved: **$3,175,350.00**

### 11. Step 8: Evaluation Metrics
Because fraud is severely imbalanced, standard accuracy is uninformative. We evaluate:

#### 1. Confusion Matrix Breakdown
- **TP** (True Positive): Fraud correctly intercepted
- **FP** (False Positive): Legitimate transaction flagged (False Alarm)
- **FN** (False Negative): Fraud missed (Direct Money Loss)
- **TN** (True Negative): Legitimate transaction approved

#### 2. Precision
Of all transactions flagged as fraud, how many were genuine fraud? High precision minimizes analyst review burden.

#### 3. Recall (Critical Metric)
Of all actual fraud occurrences, what percentage was intercepted?  
**Our Model Recall**: **99.14%**

#### 4. F2 Score
Harmonic mean giving **2x weight to Recall** ($\beta=2$) over Precision.  
**Our Model F2 Score**: **0.9840**

#### 5. ROC-AUC
Measures ranking capacity across all thresholds.  
**Our Model ROC-AUC**: **0.9914**

#### 6. PR-AUC (Precision-Recall AUC)
Gold standard for imbalanced data; evaluates precision-recall tradeoff without positive class inflation from True Negatives.

#### 7. Financial Cost Savings
$$\text{Net Saved} = (\text{Frauds Intercepted} \times \text{Average Fraud Value}) - (\text{False Positives} \times \text{Review Cost})$$  
**Net Capital Impact**: **$3,175,350.00 Saved**

### 12. Final System Output
For each inbound transaction, the deployment pipeline produces:
- Calibrated Fraud Probability Score
- Operational Action (`BLOCK`, `REVIEW`, `APPROVE`) based on the 0.10 decision threshold
- Risk Factor Attribution Matrix

---

**Executive Summary**:  
*"This enterprise system combines balance verification, behavioral modeling, network graph features, and LightGBM classification to detect financial fraud with a 99.14% recall rate and over $3.17 Million in capital loss prevention."*

---

## Data Pipeline Technical Specification

Function Signature: `preprocess_fraud_data`

### Function Overview
- Loads multi-gigabyte transaction datasets efficiently
- Generates domain feature sets (balance, temporal, behavioral, network)
- Prepares clean feature matrix $X$ and target vector $y$
- Returns optional trained `StandardScaler` for production inference

### Function Definition
```python
def preprocess_fraud_data(file_path, sample_frac=None, add_network_features=False, return_scaler=False, usecols=None):
```
Parameters:
- `file_path`: Path to input CSV (`Fraud.csv`)
- `sample_frac`: Sub-sampling fraction for fast experimentation
- `add_network_features`: Toggles recipient network graph score computation
- `return_scaler`: Returns fitted `StandardScaler` object
- `usecols`: Column selection list for memory optimization

### Memory Optimization (`dtypes`)
```python
dtypes = {
    'step': 'int32',
    'type': 'category',
    'amount': 'float32',
    'nameOrig': 'category',
    'oldbalanceOrg': 'float32',
    'newbalanceOrig': 'float32',
    'nameDest': 'category',
    'oldbalanceDest': 'float32',
    'newbalanceDest': 'float32',
    'isFraud': 'int8',
    'isFlaggedFraud': 'int8'
}
```
Impact: Standard Pandas 64-bit loading consumes 4GB+ RAM. Explicit downcasting saves **70% RAM** and speeds up reading by **3x**.

### Balance Consistency Features
```python
df['org_balance_diff'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
df['dest_balance_diff'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
```
Intuition: Normal transactions satisfy $\text{old\_balance} - \text{amount} = \text{new\_balance}$. System abuse or fraudulent overrides create non-zero delta errors.

### Relative Amount & Liquidation Features
```python
df['relative_amount'] = df['amount'] / (df['oldbalanceOrg'] + 1)
df['drained_to_zero'] = (df['newbalanceOrig'] == 0).astype('int8')
```
Intuition: Transferring $9,000 out of $10,000 (90% drain) carries significantly higher risk than $9,000 out of $1,000,000 (0.9% drain).

### Temporal Features
```python
df['hour'] = df['step'] % 24
df['is_night'] = (df['hour'] < 6).astype('int8')
```
Intuition: Off-peak transactions executed between 12 AM and 6 AM carry higher fraud odds.

### Behavioral Features
```python
df['orig_tx_count'] = df.groupby('nameOrig')['step'].transform('count')
df['orig_tx_avg_amt'] = df.groupby('nameOrig')['amount'].transform('mean')
df['dest_tx_count'] = df.groupby('nameDest')['step'].transform('count')
df['dest_tx_avg_amt'] = df.groupby('nameDest')['amount'].transform('mean')
```
Intuition: Captures individual transaction frequency and baseline spending amounts per user.

### Network Graph Topology Features
```python
if add_network_features:
    # Filter strictly for high-risk transfer mechanisms (TRANSFER and CASH_OUT)
    df_edges = df[df['type'].isin([TYPE_MAPPING.get('TRANSFER', 4), TYPE_MAPPING.get('CASH_OUT', 1)])]
    
    by_rcv = df_edges.groupby("nameDest", observed=False).agg(
        unique_senders=("nameOrig", "nunique"),
        total_received=("amount", "sum"),
        fraud_in_rate=("isFraud", "mean")
    )
```

#### Bayesian Fraud Rate Smoothing
```python
MIN_COUNT = 20
prior = df_edges['isFraud'].mean()
by_rcv['fraud_in_rate_smooth'] = (
    (by_rcv['unique_senders'] * by_rcv['fraud_in_rate'] + MIN_COUNT * prior) / 
    (by_rcv['unique_senders'] + MIN_COUNT)
)
```

#### Recipient Suspicion Rating
```python
by_rcv['suspicion_score'] = (
    by_rcv['fraud_in_rate_smooth'] * 0.7 + 
    np.log1p(by_rcv['total_received']) / 15.0 * 0.3
)
```
Intuition: Identifies high-volume mule recipient accounts receiving funds from multiple distinct senders.

---

## Detailed Intuition & Formula Reference

### 1. Sender Balance Difference (`org_balance_diff`)
- Expected Normal: $10,000 - 2,000 - 8,000 = 0$
- Fraud Scenario: $10,000 - 2,000 - 9,500 = -1,500$ (Discrepancy Detected)

### 2. Recipient Balance Difference (`dest_balance_diff`)
- Expected Normal: $5,000 + 2,000 - 7,000 = 0$
- Fraud Scenario: $5,000 + 2,000 - 5,500 = 1,500$ (Discrepancy Detected)

### 3. Relative Impact (`relative_amount`)
- High Risk: $9,000 / 10,000 = 0.90$ (90% of available capital)
- Normal Baseline: $1,000 / 100,000 = 0.01$ (1% of available capital)

---

## Performance Evaluation & Metric Matrix

### Confusion Matrix Reference
| | Predicted Normal | Predicted Fraud |
| :--- | :--- | :--- |
| **Actual Normal** | True Negative (TN) | False Positive (FP) |
| **Actual Fraud** | False Negative (FN) | True Positive (TP) |

### Performance Summary Table
| Metric | Model Performance | Strategic Significance |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9914** | High global ranking capacity |
| **Recall (Catch Rate)** | **99.14%** | Maximizes capital protection |
| **F2 Score** | **0.9840** | Prioritizes Recall over Precision |
| **Threshold Cutoff** | **0.10** | Tuned via cost matrix |
| **Net Capital Saved** | **$3,175,350.00** | Direct business impact |

---

## Top 15 Technical Interview Questions & Model Answers

### Q1: Why did you select this financial fraud project?
> **Answer**: Fraud detection represents a critical real-world engineering challenge in fintech. It requires handling large-scale data (6.36M rows), extreme class imbalance (~0.13%), custom feature engineering, and business-cost optimization.

### Q2: What makes financial fraud detection uniquely difficult?
> **Answer**: Fraud transactions are rare, highly imbalanced, dynamic, and disguised within high-volume legitimate traffic. Static threshold rules fail because fraudsters adapt, making machine learning essential.

### Q3: Why did you select LightGBM over standard decision trees or Neural Networks?
> **Answer**: LightGBM uses leaf-wise tree growth, making it significantly faster and more accurate on tabular datasets. It handles numerical and categorical features efficiently without memory overload. Our application also implements a fallback to `HistGradientBoostingClassifier` for environmental compatibility.

### Q4: Why is standard accuracy uninformative for fraud models?
> **Answer**: On a dataset with 0.13% fraud, a baseline model predicting 100% normal transactions yields 99.87% accuracy while missing 100% of actual fraud cases.

### Q5: Which primary evaluation metrics did you track?
> **Answer**: I evaluated Recall (99.14%), Precision, F2 score (0.9840), ROC-AUC (0.9914), and PR-AUC. Recall is prioritized to minimize unintercepted financial loss.

### Q6: Define Recall and explain its operational priority.
> **Answer**: Recall measures $\frac{TP}{TP + FN}$. In fraud detection, a False Negative (missed fraud) causes direct financial loss, whereas a False Positive (false alarm) incurs a minor review cost.

### Q7: Define Precision and its role in deployment.
> **Answer**: Precision measures $\frac{TP}{TP + FP}$. Maintaining strong precision prevents operational fatigue among fraud analysts and minimizes customer friction.

### Q8: What does ROC-AUC measure?
> **Answer**: ROC-AUC measures the area under the Receiver Operating Characteristic curve, representing the model's ability to rank positive fraud cases higher than normal transactions across all cutoffs.

### Q9: Why is PR-AUC preferred over ROC-AUC for imbalanced datasets?
> **Answer**: ROC-AUC can be artificially inflated by millions of True Negatives. PR-AUC evaluates precision against recall exclusively on the minority positive class.

### Q10: What was the primary objective of your feature engineering?
> **Answer**: Raw dataset fields do not explicitly indicate fraud. Feature engineering highlights balance math inconsistencies, temporal anomalies, account liquidation flags, and network hub behaviors.

### Q11: How do network graph features enhance detection?
> **Answer**: Network features analyze graph connectivity, identifying recipient hub accounts that collect funds from numerous unique senders and track incoming fraud rates across connected edges.

### Q12: How was decision threshold tuning performed?
> **Answer**: Rather than defaulting to 0.50, threshold tuning evaluated a financial cost matrix. Setting the cutoff to **0.10** maximized net dollar savings ($3,175,350.00).

### Q13: How did you prevent Data Leakage?
> **Answer**: Unique entity strings (`nameOrig`, `nameDest`) were removed prior to model fitting, ensuring the classifier learned domain behavioral patterns rather than memorizing account strings.

### Q14: What primary performance bottlenecks did you overcome?
> **Answer**: Managing 6.36M rows in memory was solved via explicit `dtypes` downcasting (saving 70% RAM), and class imbalance was handled via LightGBM loss weighting and threshold optimization.

### Q15: Describe the architecture of your production web app.
> **Answer**: Built using Streamlit and Plotly, the web app (`app.py`) provides:
> 1. Executive Operational Dashboard with dynamic KPI metrics
> 2. Real-Time Single Transaction ML Risk Predictor
> 3. Automated Batch CSV File Scanner
> 4. Interactive Recipient Network Graph Topology
> 5. Complete Technical Masterclass Documentation Guide
