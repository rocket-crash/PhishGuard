import os
import sys
import json
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
from feature_extractor import extract_features

_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_JSON_PATH = os.path.join(_DIR, 'features.json')
MODEL_OUTPUT_PATH = os.path.join(_DIR, 'model.pkl')
REPORT_OUTPUT_PATH = os.path.join(_DIR, 'model_report.json')
ACCURACY_TARGET = 0.95


# ──────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────

def load_iscx(path: str) -> pd.DataFrame:
    """
    ISCX-URL-2016: columns = [url, type]
    type ∈ {benign, phishing, malware, defacement}
    ALL non-benign types are malicious → label=1
    """
    df = pd.read_csv(path)
    df = df[['url', 'type']].copy()
    df.dropna(subset=['url', 'type'], inplace=True)
    # Normalise type to lowercase and strip whitespace
    df['type'] = df['type'].str.strip().str.lower()
    # benign=0, everything else (phishing, malware, defacement, spam)=1
    df['label'] = (df['type'] != 'benign').astype(int)
    df['source'] = 'iscx'
    return df[['url', 'label', 'source']]


def load_phishtank(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if 'url' in df.columns:
            df = df[['url']].copy()
        else:
            df = pd.DataFrame({'url': df.iloc[:, 1]})
        df.dropna(subset=['url'], inplace=True)
        df['label'] = 1
        df['source'] = 'phishtank'
        return df[['url', 'label', 'source']]
    except Exception as e:
        print(f'[!] Failed to load PhishTank: {e}')
        return pd.DataFrame(columns=['url', 'label', 'source'])


def load_urlhaus(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, comment='#', header=None,
                         names=['id', 'dateadded', 'url', 'url_status',
                                'last_online', 'threat', 'tags',
                                'urlhaus_link', 'reporter'],
                         quotechar='"')
        df = df[['url']].dropna().copy()
        df['label'] = 1
        df['source'] = 'urlhaus'
        return df[['url', 'label', 'source']]
    except Exception as e:
        print(f'[!] Failed to load URLHaus: {e}')
        return pd.DataFrame(columns=['url', 'label', 'source'])


def load_tranco(path: str, prefix: str = 'https://') -> pd.DataFrame:
    """
    Tranco Top-1M: columns = [rank, domain]
    All domains are safe → label=0
    We add realistic path/query variations to improve generalisation.
    """
    import random
    random.seed(42)
    df = pd.read_csv(path, header=None, names=['rank', 'domain'])
    urls = []
    paths = ['/', '/index.html', '/about', '/contact', '/products',
             '/login', '/search', '/help', '/privacy', '/terms']
    params = ['?id=123', '?ref=homepage', '?utm_source=internal',
              '?lang=en', '?page=1', '']
    for domain in df['domain']:
        url = prefix + domain
        if random.random() > 0.5:
            url += random.choice(paths)
            url += random.choice(params)
        urls.append(url)
    df['url'] = urls
    df['label'] = 0
    df['source'] = 'tranco'
    return df[['url', 'label', 'source']]

#combine all database

def load_and_combine(data_dir: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    iscx_path = os.path.join(data_dir, 'iscx_url_2016.csv')
    tranco_path = os.path.join(data_dir, 'tranco_top1m.csv')
    phishtank_path = os.path.join(data_dir, 'phishtank.csv')
    urlhaus_path = os.path.join(data_dir, 'urlhaus.csv')

    if os.path.isfile(iscx_path):
        print(f'[+] Loading ISCX-URL-2016 from {iscx_path}')
        iscx_df = load_iscx(iscx_path)
        print(f'    → {len(iscx_df)} rows  '
              f'(benign={len(iscx_df[iscx_df.label==0])}, '
              f'malicious={len(iscx_df[iscx_df.label==1])})')
        frames.append(iscx_df)
    else:
        print(f'[!] ISCX-URL-2016 not found at {iscx_path} — skipping')

    if os.path.isfile(phishtank_path):
        print(f'[+] Loading PhishTank from {phishtank_path}')
        pt_df = load_phishtank(phishtank_path)
        if not pt_df.empty:
            print(f'    → {len(pt_df)} malicious URLs')
            frames.append(pt_df)
    else:
        print(f'[!] PhishTank not found at {phishtank_path} — skipping')

    if os.path.isfile(urlhaus_path):
        print(f'[+] Loading URLHaus from {urlhaus_path}')
        uh_df = load_urlhaus(urlhaus_path)
        if not uh_df.empty:
            print(f'    → {len(uh_df)} malicious URLs')
            frames.append(uh_df)
    else:
        print(f'[!] URLHaus not found at {urlhaus_path} — skipping')

    if os.path.isfile(tranco_path):
        print(f'[+] Loading Tranco Top-1M from {tranco_path}')
        tranco_df = load_tranco(tranco_path)
        print(f'    → {len(tranco_df)} safe URLs')
        frames.append(tranco_df)
    else:
        print(f'[!] Tranco Top-1M not found at {tranco_path} — skipping')

    if not frames:
        print('[ERROR] No datasets found.')
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(subset='url', inplace=True)
    print(f'\n[*] Combined (deduped): {len(combined)} URLs')
    print(f'    Label distribution: '
          f'safe={len(combined[combined.label==0])}, '
          f'malicious={len(combined[combined.label==1])}')
    return combined

#EDA section

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data Preprocessing step:
    - Remove rows with null/empty URLs
    - Strip whitespace from URLs
    - Remove duplicate URLs
    - Remove rows where URL is too short to be valid
    """
    initial_count = len(df)
    print(f'\n[*] Preprocessing {initial_count} URLs …')

    # Remove nulls
    df = df.dropna(subset=['url']).copy()
    print(f'    After removing nulls: {len(df)} rows')

    # Strip whitespace and convert to string
    df['url'] = df['url'].astype(str).str.strip()

    # Remove empty strings and very short URLs
    df = df[df['url'].str.len() >= 6]
    print(f'    After removing short/empty URLs: {len(df)} rows')

    # Remove duplicates
    df = df.drop_duplicates(subset='url').reset_index(drop=True)
    print(f'    After dedup: {len(df)} rows')

    removed = initial_count - len(df)
    print(f'    Removed {removed} rows during preprocessing')
    return df

#class balancing
def balance_classes(df: pd.DataFrame, max_per_class: int = 200_000) -> pd.DataFrame:
    """
    Balance classes with controlled sampling.
    - Cap each class at max_per_class to keep training time manageable
    - Use 1:1 ratio for optimal model performance
    """
    phishing = df[df['label'] == 1]
    legit = df[df['label'] == 0]

    print(f'\n[*] Before balancing: {len(phishing)} malicious, {len(legit)} safe')

    target_count = min(len(phishing), len(legit), max_per_class)

    phishing_bal = phishing.sample(n=target_count, random_state=42)
    legit_bal = legit.sample(n=target_count, random_state=42)

    balanced = pd.concat([phishing_bal, legit_bal], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f'[*] After balancing : {target_count} malicious, {target_count} safe')
    print(f'[*] Total training samples: {len(balanced)}')
    return balanced

#feauture building
def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print('\n[*] Extracting features from URLs …')
    start = time.time()
    records: list[dict] = []
    total = len(df)
    failed = 0

    for idx, row in df.iterrows():
        try:
            feats = extract_features(row['url'])
        except Exception:
            feats = {}
            failed += 1
        feats['label'] = row['label']
        records.append(feats)
        if (idx + 1) % 20000 == 0 or idx + 1 == total:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f'    {idx + 1:>8} / {total}  ({elapsed:.1f}s, {rate:.0f} URLs/s)')

    feature_df = pd.DataFrame(records)
    feature_df.fillna(0, inplace=True)

    if failed > 0:
        print(f'[!] {failed} URLs failed feature extraction (filled with 0s)')
    print(f'[+] Feature extraction complete: {len(feature_df)} rows × '
          f'{len(feature_df.columns) - 1} features')
    return feature_df

def select_features(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """
    Feature Selection step:
    1. Remove zero/near-zero variance features (VarianceThreshold)
    2. Remove highly correlated features (correlation > 0.95)
    Returns the filtered DataFrame and the list of selected feature names.
    """
    print(f'\n[*] Feature Selection — starting with {X.shape[1]} features')


    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X)
    kept_mask = vt.get_support()
    removed_variance = [col for col, keep in zip(X.columns, kept_mask) if not keep]
    X_selected = X.loc[:, kept_mask].copy()
    if removed_variance:
        print(f'    Removed {len(removed_variance)} low-variance features: {removed_variance}')
    else:
        print(f'    No low-variance features removed')

    corr_matrix = X_selected.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    if to_drop:
        print(f'    Removed {len(to_drop)} highly correlated features: {to_drop}')
        X_selected = X_selected.drop(columns=to_drop)
    else:
        print(f'    No highly correlated features removed')

    selected_cols = list(X_selected.columns)
    print(f'[+] Feature Selection complete: {len(selected_cols)} features retained')
    return X_selected, selected_cols


def save_feature_order(columns: list[str]) -> None:
    payload = {
        '_FILE': 'features.json',
        '_PURPOSE': 'Ordered list of feature columns — MUST match model.pkl',
        '_CONNECTS_TO': 'ml/feature_extractor.py, ml/train.py',
        'features': columns
    }
    with open(FEATURES_JSON_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n[+] Feature order saved to {FEATURES_JSON_PATH}')
    print(f'    Columns ({len(columns)}): {columns}')

#printing model performence
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Evaluate a model and print all four metrics from the architecture."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f'\n  ┌─── {name} ───')
    print(f'  │  Accuracy  : {acc:.4f}')
    print(f'  │  Precision : {prec:.4f}')
    print(f'  │  Recall    : {rec:.4f}')
    print(f'  │  F1 Score  : {f1:.4f}')
    print(f'  │')
    print(f'  │  Confusion Matrix:')
    print(f'  │    TN={cm[0][0]:>6}   FP={cm[0][1]:>6}')
    print(f'  │    FN={cm[1][0]:>6}   TP={cm[1][1]:>6}')
    print(f'  └{"─" * 40}')

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': {
            'TN': int(cm[0][0]), 'FP': int(cm[0][1]),
            'FN': int(cm[1][0]), 'TP': int(cm[1][1]),
        }
    }


# Training pipeline — Ensemble (Random Forest + XGBoost + SVM)

def train(feature_df: pd.DataFrame) -> None:
    feature_cols = [c for c in feature_df.columns if c != 'label']
    X = feature_df[feature_cols]
    y = feature_df['label']

    # ── Feature Selection ──
    X_selected, selected_cols = select_features(X, y)
    save_feature_order(selected_cols)

    # ── 80/20 Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f'\n[*] Training set : {len(X_train)} samples')
    print(f'[*] Test set     : {len(X_test)} samples')
    print(f'[*] Features     : {len(selected_cols)}')

    # 1. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
    )

    # 2. XGBoost 
    xgb_model = XGBClassifier(
        n_estimators=800,
        max_depth=10,
        learning_rate=0.03,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        tree_method='hist',
        early_stopping_rounds=30,
    )

    xgb_base_params = dict(
        max_depth=10,
        learning_rate=0.03,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        tree_method='hist',
    )

    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', CalibratedClassifierCV(
            LinearSVC(
                max_iter=5000,
                random_state=42,
                C=1.0,
                class_weight='balanced',
            ),
            cv=3,
        )),
    ])
#train model indiviual

    report = {'models': {}, 'ensemble': {}, 'selected_features': selected_cols}

    # Train Random Forest
    print('\n' + '=' * 60)
    print('  TRAINING INDIVIDUAL MODELS')
    print('=' * 60)

    print('\n[*] Training Random Forest (n_estimators=500, max_depth=12) …')
    start = time.time()
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start
    print(f'[+] Random Forest trained in {rf_time:.1f}s')
    report['models']['random_forest'] = evaluate_model('Random Forest', rf_model, X_test, y_test)
    report['models']['random_forest']['train_time_s'] = round(rf_time, 1)

    # Train XGBoost
    print('\n[*] Training XGBoost (n_estimators=800, max_depth=10, lr=0.03) …')
    start = time.time()
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_time = time.time() - start
    best_n = xgb_model.best_iteration + 1  # best_iteration is 0-indexed
    print(f'[+] XGBoost trained in {xgb_time:.1f}s')
    print(f'[+] Best iteration: {xgb_model.best_iteration} → using n_estimators={best_n} for ensemble')
    report['models']['xgboost'] = evaluate_model('XGBoost', xgb_model, X_test, y_test)
    report['models']['xgboost']['train_time_s'] = round(xgb_time, 1)

    # Train SVM
    print('\n[*] Training SVM (LinearSVC + CalibratedClassifierCV, C=1.0) …')
    start = time.time()
    svm_pipeline.fit(X_train, y_train)
    svm_time = time.time() - start
    print(f'[+] SVM trained in {svm_time:.1f}s')
    report['models']['svm'] = evaluate_model('SVM (LinearSVC)', svm_pipeline, X_test, y_test)
    report['models']['svm']['train_time_s'] = round(svm_time, 1)

    # Build Ensemble — Soft Voting Classifier
    

    print('\n' + '=' * 60)
    print('  BUILDING ENSEMBLE (Soft Voting)')
    print('=' * 60)

    xgb_ensemble = XGBClassifier(n_estimators=best_n, **xgb_base_params)

    ensemble = VotingClassifier(
        estimators=[
            ('random_forest', rf_model),
            ('xgboost', xgb_ensemble),
            ('svm', svm_pipeline),
        ],
        voting='soft',
    )

    print(f'[*] Training ensemble (RF + XGBoost[n={best_n}] + SVM) …')
    start = time.time()
    ensemble.fit(X_train, y_train)

    ensemble_time = time.time() - start
    print(f'[+] Ensemble assembled in {ensemble_time:.1f}s')

    # ── Ensemble Evaluation ──
    print('\n' + '=' * 60)
    print('  ENSEMBLE EVALUATION RESULTS')
    print('=' * 60)

    report['ensemble'] = evaluate_model('ENSEMBLE (RF + XGBoost + SVM)', ensemble, X_test, y_test)
    report['ensemble']['train_time_s'] = round(ensemble_time, 1)

    # Full classification report
    y_pred_ensemble = ensemble.predict(X_test)
    print('\n' + classification_report(y_test, y_pred_ensemble, target_names=['legit', 'phishing']))

    # ── Feature importance (from Random Forest component) ──
    rf_importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': selected_cols,
        'importance': rf_importances
    }).sort_values('importance', ascending=False)
    print('  Top 15 Feature Importances (from Random Forest):')
    print('  ' + '-' * 45)
    for _, row in importance_df.head(15).iterrows():
        bar = '█' * int(row['importance'] * 100)
        print(f'    {row["feature"]:30s}  {row["importance"]:.4f}  {bar}')
    print()

    # ── Inference speed test ──
    print('[*] Running inference speed test (1000 predictions) …')
    speed_X = X_test.head(1000) if len(X_test) >= 1000 else X_test
    speed_start = time.time()
    for _ in range(10):
        _ = ensemble.predict(speed_X)
    speed_elapsed = (time.time() - speed_start) / 10
    per_sample_ms = (speed_elapsed / len(speed_X)) * 1000
    print(f'  Batch of {len(speed_X)}: {speed_elapsed*1000:.1f} ms total')
    print(f'  Per-sample inference: {per_sample_ms:.3f} ms')
    report['inference_speed'] = {
        'batch_size': len(speed_X),
        'batch_ms': round(speed_elapsed * 1000, 1),
        'per_sample_ms': round(per_sample_ms, 3),
    }
    print()

    acc = report['ensemble']['accuracy']
    if acc >= ACCURACY_TARGET:
        print(f'[PASS] Ensemble accuracy {acc:.4f} meets the >={ACCURACY_TARGET} target!')
    else:
        print(f'[FAIL] Ensemble accuracy {acc:.4f} is BELOW the >={ACCURACY_TARGET} target.')
        print()
        print('  SUGGESTIONS TO IMPROVE:')
        print('  1. Add more training data — especially diverse phishing samples')
        print('  2. Engineer more features — e.g. WHOIS age, page-content similarity')
        print('  3. Tune hyperparameters for individual models')
        print('  4. Try different ensemble weights')
        print('  5. Inspect confusion matrix: if FN is high, add more phishing examples')

    joblib.dump(ensemble, MODEL_OUTPUT_PATH)
    print(f'\n[+] Ensemble model saved to {MODEL_OUTPUT_PATH}')
    with open(REPORT_OUTPUT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[+] Evaluation report saved to {REPORT_OUTPUT_PATH}')


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_DIR, 'data')

    print('=' * 60)
    print('  PhishGuard — ML Training Pipeline v3')
    print('  Ensemble: Random Forest + XGBoost + SVM')
    print('  35 features · Feature Selection · 4 datasets')
    print('=' * 60)
    raw_df = load_and_combine(data_dir)
    clean_df = preprocess_data(raw_df)
    balanced_df = balance_classes(clean_df)
    feature_df = build_feature_dataframe(balanced_df)

    train(feature_df)

    print('\n[DONE] Pipeline complete.')


if __name__ == '__main__':
    main()