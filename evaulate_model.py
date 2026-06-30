import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import pickle
import os

def evaluate():
    print("=" * 60)
    print("📊 EVALUATION METRICS (For Thesis)")
    print("=" * 60)
    
    if not os.path.exists('dataset_features.csv'):
        print("❌ dataset_features.csv not found!")
        print("📋 Run 'python create_dataset.py' first")
        return
    
    df = pd.read_csv('dataset_features.csv')
    print(f"📊 Total samples: {len(df)}")
    print(f"   Genuine (0): {sum(df['label']==0)}")
    print(f"   Fake (1): {sum(df['label']==1)}")
    
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if os.path.exists('fraud_model.pkl'):
        with open('fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded: fraud_model.pkl")
    else:
        print("🚀 Training new model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        with open('fraud_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("✅ Model saved: fraud_model.pkl")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print("\n" + "=" * 60)
    print("📋 THESIS EVALUATION RESULTS")
    print("=" * 60)
    print(f"✅ Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Precision:    {precision:.4f} ({precision*100:.2f}%)  ← H1: Target > 85%")
    print(f"✅ Recall:       {recall:.4f} ({recall*100:.2f}%)       ← H1: Target > 80%")
    print(f"✅ F1 Score:     {f1:.4f} ({f1*100:.2f}%)")
    print(f"✅ FPR:          {fpr:.4f} ({fpr*100:.2f}%)             ← H2: Target < 5%")
    print("=" * 60)
    
    print("\n📊 CONFUSION MATRIX:")
    print(f"   True Positives:  {tp}  (Correctly identified fake)")
    print(f"   True Negatives:  {tn}  (Correctly identified genuine)")
    print(f"   False Positives: {fp}  (Genuine flagged as fake) ← False Accusations")
    print(f"   False Negatives: {fn}  (Fake missed)")
    print("=" * 60)
    
    print("\n📋 HYPOTHESIS VERIFICATION:")
    print("-" * 40)
    h1_precision = "✅ PASSED" if precision >= 0.85 else "❌ FAILED"
    h1_recall = "✅ PASSED" if recall >= 0.80 else "❌ FAILED"
    h2_fpr = "✅ PASSED" if fpr < 0.05 else "❌ FAILED"
    
    print(f"📌 H1 (Precision > 85%): {h1_precision} ({precision*100:.1f}%)")
    print(f"📌 H1 (Recall > 80%):    {h1_recall} ({recall*100:.1f}%)")
    print(f"📌 H2 (FPR < 5%):        {h2_fpr} ({fpr*100:.1f}%)")
    print("-" * 40)

if __name__ == '__main__':
    evaluate()