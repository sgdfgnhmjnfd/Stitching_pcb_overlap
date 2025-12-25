"""
Analyze data distribution to find natural thresholds
And retrain model with better balanced labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def analyze_distribution():
    """Analyze PSNR distribution to find natural breaks"""
    metrics_file = Path(r"d:\Dataset_PCB_Final\results\evaluation\ALL_METRICS.csv")
    df = pd.read_csv(metrics_file)
    
    print("\n" + "="*80)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print(f"\nPSNR_FINAL Statistics:")
    print(f"  Min: {df['PSNR_FINAL'].min():.2f}")
    print(f"  25%: {df['PSNR_FINAL'].quantile(0.25):.2f}")
    print(f"  Median: {df['PSNR_FINAL'].median():.2f}")
    print(f"  75%: {df['PSNR_FINAL'].quantile(0.75):.2f}")
    print(f"  Max: {df['PSNR_FINAL'].max():.2f}")
    print(f"  Mean: {df['PSNR_FINAL'].mean():.2f}")
    print(f"  Std: {df['PSNR_FINAL'].std():.2f}")
    
    print(f"\nSSIM_FINAL Statistics:")
    print(f"  Min: {df['SSIM_FINAL'].min():.4f}")
    print(f"  25%: {df['SSIM_FINAL'].quantile(0.25):.4f}")
    print(f"  Median: {df['SSIM_FINAL'].median():.4f}")
    print(f"  75%: {df['SSIM_FINAL'].quantile(0.75):.4f}")
    print(f"  Max: {df['SSIM_FINAL'].max():.4f}")
    print(f"  Mean: {df['SSIM_FINAL'].mean():.4f}")
    print(f"  Std: {df['SSIM_FINAL'].std():.4f}")
    
    # Find natural breaks using quantiles
    p25 = df['PSNR_FINAL'].quantile(0.25)
    p50 = df['PSNR_FINAL'].quantile(0.50)
    p75 = df['PSNR_FINAL'].quantile(0.75)
    
    print(f"\n✓ Suggested Natural Thresholds:")
    print(f"  Poor (bottom 25%): PSNR < {p25:.2f}")
    print(f"  Fair (middle 50%): {p25:.2f} <= PSNR < {p75:.2f}")
    print(f"  Good (top 25%): PSNR >= {p75:.2f}")
    
    return p25, p75, df

def create_balanced_labels(df, p25, p75):
    """Create labels based on natural distribution"""
    df['label'] = 'fair'
    df.loc[df['PSNR_FINAL'] < p25, 'label'] = 'poor'
    df.loc[df['PSNR_FINAL'] >= p75, 'label'] = 'good'
    
    print(f"\n\nLabel Distribution (Natural Quantiles):")
    print(df['label'].value_counts())
    return df

def train_balanced_model(df):
    """Train model with better data balance"""
    print("\n" + "="*80)
    print("TRAINING BALANCED MODEL")
    print("="*80)
    
    # Prepare features
    features = ['PSNR_STITCH', 'SSIM_STITCH', 'IOU_STITCH', 'PSNR_FINAL', 'SSIM_FINAL', 'IOU_FINAL']
    X = df[features].copy()
    y = df['label'].copy()
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\n✓ Classes: {le.classes_}")
    print(f"✓ Feature shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n✓ Train size: {len(X_train)}")
    print(f"✓ Test size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with class weights to handle imbalance
    print("\n[TRAINING] LightGBM with balanced class weights...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*80}")
    print(f"REGULAR ACCURACY:  {accuracy:.2%}")
    print(f"BALANCED ACCURACY: {balanced_acc:.2%}")
    print(f"{'='*80}")
    
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nCONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model
    model_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\lgb_model_balanced.joblib")
    scaler_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\feature_scaler_balanced.joblib")
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, model_path.parent / "label_encoder_balanced.joblib")
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Feature importance
    print("\nFEATURE IMPORTANCE:")
    for name, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")
    
    return model, scaler, le

def main():
    print("\n" + "="*80)
    print("RETRAINING MODEL WITH BETTER THRESHOLDS")
    print("="*80)
    
    # Analyze distribution
    p25, p75, df = analyze_distribution()
    
    # Create balanced labels
    df = create_balanced_labels(df, p25, p75)
    
    # Train model
    train_balanced_model(df)
    
    print("\n" + "="*80)
    print("✓ DONE - Model trained with natural quantile-based thresholds")
    print("="*80)

if __name__ == "__main__":
    main()
