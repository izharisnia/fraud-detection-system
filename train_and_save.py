import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from preprocessing import preprocess_fraud_data, save_artifacts

# Try importing LightGBM, fallback to HistGradientBoostingClassifier if C++ libomp missing
try:
    import lightgbm as lgb
    USE_LGBM = True
    print("Using LightGBM Classifier engine.")
except Exception as e:
    from sklearn.ensemble import HistGradientBoostingClassifier
    USE_LGBM = False
    print(f"LightGBM not available ({e}). Using Sklearn HistGradientBoostingClassifier engine.")

def train_and_export(file_path="Fraud.csv", sample_frac=0.2):
    print(f"Loading and preprocessing {file_path} (sample_frac={sample_frac})...")
    start_time = time.time()
    
    X, y, scaler, _ = preprocess_fraud_data(
        file_path_or_df=file_path,
        sample_frac=sample_frac,
        add_network_features=True,
        return_scaler=True
    )
    
    feature_columns = list(X.columns)
    print(f"Dataset preprocessed in {time.time() - start_time:.2f}s. Shape: {X.shape}, Fraud rate: {y.mean():.4f}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training Fraud Classifier model...")
    if USE_LGBM:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'is_unbalance': True,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=150,
            valid_sets=[valid_data]
        )
        y_pred_prob = model.predict(X_valid)
        model_name = "LightGBM Classifier"
    else:
        model = HistGradientBoostingClassifier(
            class_weight='balanced',
            max_iter=150,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_valid)[:, 1]
        model_name = "HistGradientBoosting Classifier"

    auc_score = roc_auc_score(y_valid, y_pred_prob)
    print(f"Model Training Complete! ({model_name}) Validation ROC-AUC: {auc_score:.4f}")

    # Threshold tuning based on economic cost (Fraud savings vs False alarm penalty)
    precisions, recalls, thresholds = precision_recall_curve(y_valid, y_pred_prob)
    best_t = 0.5
    max_savings = -float('inf')

    fraud_val = 10000
    fa_cost = 50

    for t in np.linspace(0.1, 0.9, 81):
        preds = (y_pred_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_valid == 1))
        fp = np.sum((preds == 1) & (y_valid == 0))
        savings = (tp * fraud_val) - (fp * fa_cost)
        if savings > max_savings:
            max_savings = savings
            best_t = float(t)

    print(f"Optimal Threshold: {best_t:.4f} with estimated savings: ${max_savings:,.2f}")

    metadata = {
        "model_type": model_name,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_threshold": round(best_t, 4),
        "roc_auc": round(auc_score, 4),
        "feature_columns": feature_columns,
        "sample_frac": sample_frac
    }

    save_artifacts(model, scaler, metadata, output_dir=".")
    print("All training artifacts generated successfully.")

if __name__ == "__main__":
    train_and_export()
