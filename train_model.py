import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
warnings.filterwarnings('ignore')

FEATURE_COLS = ['amount_present','txnid_present','status_positive','status_present',
    'platform_detected','date_present','text_length','word_count','digit_ratio','amount_repeat']

def load_data():
    if os.path.exists('payment_features.csv'):
        df = pd.read_csv('payment_features.csv')
        print(f"[+] Loaded {len(df)} samples — Genuine: {(df.label==0).sum()} | Fake: {(df.label==1).sum()}")
        return df
    print("[!] payment_features.csv not found. Run create_dataset.py first.\n    Using synthetic demo data...\n")
    np.random.seed(42); n=200
    genuine = dict(amount_present=np.ones(n//2), txnid_present=np.random.choice([1,1,0],n//2),
        status_positive=np.random.choice([1,1,1,0],n//2), status_present=np.ones(n//2),
        platform_detected=np.random.choice([1,1,0],n//2), date_present=np.random.choice([1,0],n//2),
        text_length=np.random.uniform(0.4,1.0,n//2), word_count=np.random.uniform(0.3,0.9,n//2),
        digit_ratio=np.random.uniform(0.05,0.2,n//2), amount_repeat=np.random.choice([1,2],n//2),
        label=np.zeros(n//2,dtype=int))
    fake = dict(amount_present=np.random.choice([1,0],n//2), txnid_present=np.random.choice([0,1],n//2),
        status_positive=np.random.choice([0,1],n//2), status_present=np.random.choice([0,1],n//2),
        platform_detected=np.random.choice([0,1],n//2), date_present=np.random.choice([0,1],n//2),
        text_length=np.random.uniform(0.0,0.5,n//2), word_count=np.random.uniform(0.0,0.4,n//2),
        digit_ratio=np.random.uniform(0.0,0.1,n//2), amount_repeat=np.random.choice([0,3,4],n//2),
        label=np.ones(n//2,dtype=int))
    return pd.concat([pd.DataFrame(genuine), pd.DataFrame(fake)]).sample(frac=1, random_state=42)

def train():
    df = load_data()
    X, y = df[FEATURE_COLS].values, df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n" + "="*50 + "\n  RESULTS\n" + "="*50)
    print(f"  Precision : {precision_score(y_test,y_pred,zero_division=0):.3f}  (target >= 0.85)")
    print(f"  Recall    : {recall_score(y_test,y_pred,zero_division=0):.3f}  (target >= 0.80)")
    print(f"  F1 Score  : {f1_score(y_test,y_pred,zero_division=0):.3f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:\n    TN={cm[0,0]}  FP={cm[0,1]}\n    FN={cm[1,0]}  TP={cm[1,1]}")
    cv = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"\n  5-Fold CV F1: {cv.mean():.3f} +/- {cv.std():.3f}\n" + "="*50)
    with open('fraud_model.pkl','wb') as f: pickle.dump(model, f)
    print("\n[OK] Model saved: fraud_model.pkl\n[OK] Restart app.py to use the trained model\n")

if __name__ == '__main__':
    train()
