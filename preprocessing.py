import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

TYPE_MAPPING = {
    'CASH_IN': 0,
    'CASH_OUT': 1,
    'DEBIT': 2,
    'PAYMENT': 3,
    'TRANSFER': 4
}

def preprocess_fraud_data(file_path_or_df, sample_frac=None, add_network_features=True, return_scaler=True):
    """
    Preprocess financial transactions dataset for Fraud Detection.
    Handles data loading, feature engineering, network analytics, and scaling.
    """
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
    
    if isinstance(file_path_or_df, str):
        df = pd.read_csv(file_path_or_df, dtype=dtypes)
    else:
        df = file_path_or_df.copy()
        
    if sample_frac is not None and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # 1. Feature Engineering
    df['org_balance_diff'] = (df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']).astype('float32')
    df['dest_balance_diff'] = (df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']).astype('float32')
    df['relative_amount'] = (df['amount'] / (df['oldbalanceOrg'] + 1.0)).astype('float32')
    df['drained_to_zero'] = (df['newbalanceOrig'] == 0).astype('int8')
    df['hour'] = (df['step'] % 24).astype('int32')
    df['is_night'] = (df['hour'] < 6).astype('int8')

    # Sender & Receiver history aggregations
    df['orig_tx_count'] = df.groupby('nameOrig', observed=False)['step'].transform('count').astype('int32')
    df['orig_tx_avg_amt'] = df.groupby('nameOrig', observed=False)['amount'].transform('mean').astype('float32')
    df['dest_tx_count'] = df.groupby('nameDest', observed=False)['step'].transform('count').astype('int32')
    df['dest_tx_avg_amt'] = df.groupby('nameDest', observed=False)['amount'].transform('mean').astype('float32')
    df['is_merchant'] = df['nameDest'].astype(str).str.startswith('M').astype('int8')

    # Strictly encode categorical 'type' as integer
    df['type'] = df['type'].astype(str).map(TYPE_MAPPING).fillna(4).astype('int8')

    # 2. Network Graph Features
    if add_network_features:
        df_edges = df[["nameOrig", "nameDest", "amount", "isFraud", "step"]].copy()
        by_rcv = df_edges.groupby("nameDest", observed=False).agg(
            unique_senders=("nameOrig", "nunique"),
            total_received=("amount", "sum"),
            fraud_in_rate=("isFraud", "mean")
        ).reset_index()

        MIN_COUNT = 20
        prior = float(df_edges['isFraud'].mean()) if 'isFraud' in df_edges else 0.001
        by_rcv['fraud_in_rate_smooth'] = ((by_rcv['fraud_in_rate'] * by_rcv['unique_senders'] + prior * MIN_COUNT) /
                                          (by_rcv['unique_senders'] + MIN_COUNT)).astype('float32')
        max_tot = by_rcv['total_received'].max()
        max_tot_log = np.log1p(max_tot) if max_tot > 0 else 1.0
        by_rcv['suspicion_score'] = (0.5 * by_rcv['fraud_in_rate_smooth'] +
                                     0.5 * (np.log1p(by_rcv['total_received']) / max_tot_log)).astype('float32')

        df = df.merge(by_rcv[['nameDest', 'unique_senders', 'total_received', 'fraud_in_rate_smooth', 'suspicion_score']],
                      how='left', on='nameDest')
        
        for col in ['unique_senders', 'total_received', 'fraud_in_rate_smooth', 'suspicion_score']:
            df[col] = df[col].fillna(0).astype('float32')

    df_with_ids = df.copy()
    df = df.drop(columns=['nameOrig', 'nameDest'], errors='ignore')

    if 'isFraud' in df.columns:
        y = df['isFraud']
        X = df.drop(columns=['isFraud', 'isFlaggedFraud'], errors='ignore')
    else:
        y = None
        X = df.drop(columns=['isFlaggedFraud'], errors='ignore')

    scaler = None
    if return_scaler:
        scaler = StandardScaler()
        # Scale continuous float columns
        float_cols = X.select_dtypes(include=['float32', 'float64']).columns
        X[float_cols] = scaler.fit_transform(X[float_cols])

    return X, y, scaler, df_with_ids

def preprocess_single_transaction(input_dict, feature_columns, scaler=None):
    """
    Preprocess a single raw transaction input into the feature format expected by the model.
    """
    step = int(input_dict.get('step', 1))
    tx_type = str(input_dict.get('type', 'TRANSFER')).upper()
    amount = float(input_dict.get('amount', 0.0))
    oldbalanceOrg = float(input_dict.get('oldbalanceOrg', 0.0))
    newbalanceOrig = float(input_dict.get('newbalanceOrig', 0.0))
    oldbalanceDest = float(input_dict.get('oldbalanceDest', 0.0))
    newbalanceDest = float(input_dict.get('newbalanceDest', 0.0))
    nameDest = str(input_dict.get('nameDest', 'M12345678'))

    type_code = TYPE_MAPPING.get(tx_type, 4)
    org_balance_diff = oldbalanceOrg - amount - newbalanceOrig
    dest_balance_diff = oldbalanceDest + amount - newbalanceDest
    relative_amount = amount / (oldbalanceOrg + 1.0)
    drained_to_zero = 1 if newbalanceOrig == 0 else 0
    hour = step % 24
    is_night = 1 if hour < 6 else 0
    is_merchant = 1 if nameDest.startswith('M') else 0

    is_high_risk_type = 1 if tx_type in ['TRANSFER', 'CASH_OUT'] else 0
    if is_high_risk_type and (drained_to_zero or relative_amount > 0.8):
        fraud_in_rate_smooth = 1.0
        suspicion_score = 0.6614
    elif is_high_risk_type:
        fraud_in_rate_smooth = 0.15
        suspicion_score = 0.45
    else:
        fraud_in_rate_smooth = 0.001
        suspicion_score = 0.1 if is_merchant else 0.2

    feature_dict = {
        'step': np.int32(step),
        'type': np.int8(type_code),
        'amount': np.float32(amount),
        'oldbalanceOrg': np.float32(oldbalanceOrg),
        'newbalanceOrig': np.float32(newbalanceOrig),
        'oldbalanceDest': np.float32(oldbalanceDest),
        'newbalanceDest': np.float32(newbalanceDest),
        'org_balance_diff': np.float32(org_balance_diff),
        'dest_balance_diff': np.float32(dest_balance_diff),
        'relative_amount': np.float32(relative_amount),
        'drained_to_zero': np.int8(drained_to_zero),
        'hour': np.int32(hour),
        'is_night': np.int8(is_night),
        'orig_tx_count': np.int32(1),
        'orig_tx_avg_amt': np.float32(amount),
        'dest_tx_count': np.int32(1),
        'dest_tx_avg_amt': np.float32(amount),
        'is_merchant': np.int8(is_merchant),
        'unique_senders': np.float32(1.0),
        'total_received': np.float32(amount),
        'fraud_in_rate_smooth': np.float32(fraud_in_rate_smooth),
        'suspicion_score': np.float32(suspicion_score)
    }

    df_single = pd.DataFrame([feature_dict])
    df_single = df_single.reindex(columns=feature_columns, fill_value=0)

    if scaler is not None and hasattr(scaler, 'feature_names_in_'):
        scale_cols = list(scaler.feature_names_in_)
        df_single[scale_cols] = scaler.transform(df_single[scale_cols])

    return df_single

def save_artifacts(model, scaler, metadata, output_dir="."):
    """Save trained model, scaler, and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "fraud_model.pkl"))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Artifacts successfully saved to {output_dir}")

def load_artifacts(artifact_dir="."):
    """Load model, scaler, and metadata from disk."""
    model_path = os.path.join(artifact_dir, "fraud_model.pkl")
    scaler_path = os.path.join(artifact_dir, "scaler.pkl")
    metadata_path = os.path.join(artifact_dir, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return model, scaler, metadata
