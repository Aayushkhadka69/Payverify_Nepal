import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pickle
import os

def train_model():
    """Train the ML model for fraud detection"""
    
    print("=" * 60)
    print("🤖 TRAINING MACHINE LEARNING MODEL")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists('dataset_features.csv'):
        print("❌ dataset_features.csv not found!")
        print("📋 Run 'python create_dataset.py' first")
        return
    
    # Load dataset
    df = pd.read_csv('dataset_features.csv')
    print(f"📊 Dataset loaded: {len(df)} samples")
    print(f"   Genuine (0): {sum(df['label']==0)}")
    print(f"   Fake (1): {sum(df['label']==1)}")
    
    # Split features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train Random Forest
    print("\n🚀 Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Save model
    with open('fraud_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Precision: {precision:.4f} ({precision*100:.2f}%)  ← H1: Target > 85%")
    print(f"✅ Recall:    {recall:.4f} ({recall*100:.2f}%)       ← H1: Target > 80%")
    print(f"✅ F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"✅ FPR:       {fpr:.4f} ({fpr*100:.2f}%)             ← H2: Target < 5%")
    print("=" * 60)
    
    # H1 Check
    print("\n📋 HYPOTHESIS CHECKS:")
    print("-" * 40)
    if precision >= 0.85:
        print("✅ H1 (Precision > 85%): PASSED")
    else:
        print(f"❌ H1 (Precision > 85%): FAILED ({precision*100:.1f}%)")
    
    if recall >= 0.80:
        print("✅ H1 (Recall > 80%): PASSED")
    else:
        print(f"❌ H1 (Recall > 80%): FAILED ({recall*100:.1f}%)")
    
    if fpr < 0.05:
        print("✅ H2 (FPR < 5%): PASSED")
    else:
        print(f"❌ H2 (FPR < 5%): FAILED ({fpr*100:.1f}%)")
    print("-" * 40)
    
    # Feature importance
    feature_names = X.columns
    importances = model.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
    print("\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
    for name, score in top_features:
        print(f"   {name}: {score:.4f}")
    
    print("\n📁 Model saved as: fraud_model.pkl")
    print("=" * 60)
    
    return model, precision, recall, f1, fpr

if __name__ == '__main__':
    train_model()