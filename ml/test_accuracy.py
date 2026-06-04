import os
import sys
import json
import time
import pandas as pd
import numpy as np
import joblib
from multiprocessing import Pool
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# Add current dir to sys.path to allow importing feature_extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features

DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_JSON_PATH = os.path.join(DIR, 'features.json')
MODEL_PATH = os.path.join(DIR, 'model.pkl')
TEST_DATA_PATH = os.path.join(DIR, '..', 'test', 'new_data_urls.csv')


def process_chunk(chunk):
    records = []
    for idx, row in chunk.iterrows():
        try:
            feats = extract_features(str(row['url']))
        except Exception:
            feats = {}
        feats['status'] = row['status']
        records.append(feats)
    return pd.DataFrame(records)


def main():
    print(f"[*] Loading ensemble model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    with open(FEATURES_JSON_PATH, 'r') as f:
        feature_cols = json.load(f)['features']

    print(f"[*] Loading test data from {TEST_DATA_PATH}...")
    df = pd.read_csv(TEST_DATA_PATH)

    # In test/new_data_urls.csv, 0 is phishing and 1 is benign.
    # In our model, 0 is benign and 1 is phishing.
    # So we must invert the labels!
    df['status'] = 1 - df['status']

    print(f"[*] Extracting features for {len(df)} URLs...")

    num_cores = os.cpu_count() or 1
    chunks = np.array_split(df, num_cores)

    start = time.time()
    with Pool(num_cores) as pool:
        result_dfs = pool.map(process_chunk, chunks)

    feature_df = pd.concat(result_dfs, ignore_index=True)
    feature_df.fillna(0, inplace=True)

    elapsed = time.time() - start
    print(f"[*] Feature extraction completed in {elapsed:.1f}s.")

    # Ensure columns match
    for col in feature_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0

    X = feature_df[feature_cols]
    y_true = feature_df['status']

    print("[*] Predicting with ensemble model...")
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print('\n' + '=' * 60)
    print('  EVALUATION RESULTS ON TEST DATA')
    print('  Model: Ensemble (Random Forest + XGBoost + SVM)')
    print('=' * 60)
    print(f'  Accuracy  : {acc:.4f}')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  F1 Score  : {f1:.4f}')
    print()
    print('  Confusion Matrix:')
    if cm.shape == (2, 2):
        print(f'    TN={cm[0][0]:>6}   FP={cm[0][1]:>6}')
        print(f'    FN={cm[1][0]:>6}   TP={cm[1][1]:>6}')
    else:
        print(cm)
    print()
    print(classification_report(y_true, y_pred, target_names=['legit', 'phishing']))

    # Inference speed test
    print('[*] Running inference speed test …')
    speed_X = X.head(1000) if len(X) >= 1000 else X
    speed_start = time.time()
    for _ in range(10):
        _ = model.predict(speed_X)
    speed_elapsed = (time.time() - speed_start) / 10
    per_sample_ms = (speed_elapsed / len(speed_X)) * 1000
    print(f'  Batch of {len(speed_X)}: {speed_elapsed*1000:.1f} ms total')
    print(f'  Per-sample inference: {per_sample_ms:.3f} ms')


if __name__ == '__main__':
    main()
