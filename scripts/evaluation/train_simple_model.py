"""
Simple training script for current metrics data
Works with PSNR_STITCH, SSIM_STITCH, IOU_STITCH, PSNR_FINAL, SSIM_FINAL, IOU_FINAL
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_metrics():
    """Load ALL_METRICS.csv"""
    metrics_file = Path(r"d:\Dataset_PCB_Final\results\evaluation\ALL_METRICS.csv")
    df = pd.read_csv(metrics_file)
    print(f"✓ Loaded {len(df)} samples")
    print(f"✓ Columns: {list(df.columns)}")
    return df

def create_labels(df):
    """Create quality labels based on PSNR_FINAL"""
    # Define thresholds
    df['label'] = 'good'
    df.loc[df['PSNR_FINAL'] < 15, 'label'] = 'poor'
    df.loc[(df['PSNR_FINAL'] >= 15) & (df['PSNR_FINAL'] < 18), 'label'] = 'fair'
    
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    return df

def prepare_features(df):
    """Use available metrics as features"""
    features = ['PSNR_STITCH', 'SSIM_STITCH', 'IOU_STITCH', 'PSNR_FINAL', 'SSIM_FINAL', 'IOU_FINAL']
    
    X = df[features].copy()
    y = df['label'].copy()
    
    print(f"\n✓ Features: {features}")
    print(f"✓ Feature shape: {X.shape}")
    
    return X, y, features

def train_model():
    """Train LightGBM model"""
    print("\n" + "="*80)
    print("TRAINING SIMPLE MODEL")
    print("="*80)
    
    # Load and prepare data
    df = load_metrics()
    df = create_labels(df)
    X, y, feature_names = prepare_features(df)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\n✓ Classes: {le.classes_}")
    
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
    
    # Train LightGBM
    print("\n[TRAINING] LightGBM Model...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=5,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*80}")
    print(f"ACCURACY: {accuracy:.2%}")
    print(f"{'='*80}")
    
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nCONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model and scaler
    model_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\lgb_model_simple.joblib")
    scaler_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\feature_scaler_simple.joblib")
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Feature importance
    print("\nFEATURE IMPORTANCE:")
    for name, imp in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {imp:.4f}")

if __name__ == "__main__":
    train_model()
