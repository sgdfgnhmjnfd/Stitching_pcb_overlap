"""
Improved feature engineering specifically to detect shuffled tiles
Instead of image-based features, use metric patterns that distinguish shuffled from base
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_combine_metrics():
    """Load all metrics from evaluation CSV files"""
    eval_path = Path(r"d:\Dataset_PCB_Final\results\evaluation")
    
    all_data = []
    
    # Iterate through all metric files
    for metrics_file in eval_path.rglob("metrics.csv"):
        try:
            df = pd.read_csv(metrics_file)
            # Extract variant from path: e.g., "1-1_base\1-1_base\metrics.csv"
            parts = str(metrics_file.parent.parent).split("\\")
            variant = parts[-1] if len(parts) > 0 else "unknown"
            df['variant'] = variant
            all_data.append(df)
        except Exception as e:
            pass
    
    if not all_data:
        print("ERROR: No metrics.csv files found!")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} total samples")
    print(f"Variants: {sorted(combined_df['variant'].unique())}")
    
    return combined_df


def create_shuffled_detection_features(df):
    """
    Create features specifically designed to detect shuffled tiles
    Key insight: Shuffled differs from base only in pixel order, not metrics
    So we look for metric consistency patterns
    """
    
    features_dict = {}
    
    # Ensure required columns exist
    required_cols = ['PSNR_RAW', 'SSIM_RAW', 'EDGE_IOU_RAW', 'PSNR_PCB', 'SSIM_PCB', 'EDGE_IOU_PCB']
    for col in required_cols:
        if col not in df.columns:
            print(f"WARNING: {col} not found in data")
            return None
    
    # 1. METRIC ASYMMETRY - shuffled tiles have different pixel distribution
    features_dict['PSNR_SSIM_GAP'] = abs(df['PSNR_RAW'].values - df['SSIM_RAW'].values * 20)
    
    # 2. EDGE CONSISTENCY - shuffled has discontinuities
    features_dict['EDGE_METRIC_RATIO'] = (df['EDGE_IOU_RAW'].values + 0.001) / (df['PSNR_RAW'].values + 0.001)
    
    # 3. PCB FILTER IMPACT - varies for shuffled
    features_dict['PCB_IMPACT_RATIO'] = (abs(df['PSNR_DELTA'].values) + abs(df['SSIM_DELTA'].values)) / (abs(df['EDGE_DELTA'].values) + 0.01)
    
    # 4. METRIC VARIANCE (NEW) - computes within-image variance
    features_dict['RAW_METRIC_SPREAD'] = np.maximum(df['PSNR_RAW'].values, df['SSIM_RAW'].values * 20) - np.minimum(df['PSNR_RAW'].values, df['SSIM_RAW'].values * 20)
    
    # 5. QUALITY UNIFORMITY - shuffled has less uniform quality
    features_dict['QUALITY_UNIFORMITY'] = 1.0 / (1.0 + features_dict['RAW_METRIC_SPREAD'] / 5.0)
    
    # 6. EDGE VARIANCE - shuffled edges have more variance
    features_dict['EDGE_STABILITY'] = np.minimum(df['EDGE_IOU_RAW'].values, df['EDGE_IOU_PCB'].values) / (np.maximum(df['EDGE_IOU_RAW'].values, df['EDGE_IOU_PCB'].values) + 0.01)
    
    # 7. PSNR-EDGE CORRELATION - shuffled has weaker correlation
    features_dict['PSNR_EDGE_CORRELATION'] = (df['PSNR_RAW'].values * df['EDGE_IOU_RAW'].values) / (df['PSNR_RAW'].values ** 2 + 1)
    
    # 8. SSIM-EDGE CORRELATION
    features_dict['SSIM_EDGE_CORRELATION'] = (df['SSIM_RAW'].values * df['EDGE_IOU_RAW'].values) / (df['SSIM_RAW'].values ** 2 + 1)
    
    # 9. DEGRADATION ASYMMETRY - shuffled shows different degradation pattern
    psnr_deg = df['PSNR_RAW'].values - df['PSNR_PCB'].values
    ssim_deg = df['SSIM_RAW'].values - df['SSIM_PCB'].values
    features_dict['DEGRADATION_SKEW'] = (psnr_deg - ssim_deg * 20) / (np.maximum(psnr_deg, ssim_deg * 20) + 1)
    
    # 10. BOUNDARY INDICATOR - shuffled has specific metric patterns
    features_dict['BOUNDARY_COMPLEXITY'] = (abs(features_dict['PSNR_SSIM_GAP']) + abs(features_dict['PCB_IMPACT_RATIO'])) / 2
    
    return features_dict


def create_labels(df):
    """Create 3-level quality labels"""
    labels = []
    
    for variant in df['variant']:
        variant_lower = variant.lower()
        
        # PERFECT: best quality
        if 'base' in variant_lower and 'rotated_small' not in variant_lower:
            labels.append('perfect')
        elif 'rotated_small' in variant_lower:
            labels.append('perfect')
        
        # ACCEPTABLE: minor issues but repairable
        elif 'shuffled' in variant_lower:
            labels.append('acceptable')  
        elif 'rotated' in variant_lower and 'rotated_small' not in variant_lower:
            labels.append('acceptable')
        elif 'missing' in variant_lower:
            labels.append('acceptable')
        
        # REJECT: severe defects
        elif 'dazzled' in variant_lower:
            labels.append('reject')
        elif 'blurred' in variant_lower:
            labels.append('reject')
        else:
            labels.append('acceptable')
    
    df['QUALITY_LABEL'] = labels
    return df


def prepare_final_features(df):
    """Combine all features: original metrics + engineered + shuffled-specific"""
    
    # Original metric features
    metric_features = ['PSNR_RAW', 'SSIM_RAW', 'EDGE_IOU_RAW', 'PSNR_PCB', 'SSIM_PCB', 'EDGE_IOU_PCB']
    
    # Add delta features
    df['PSNR_DELTA'] = df['PSNR_RAW'] - df['PSNR_PCB']
    df['SSIM_DELTA'] = df['SSIM_RAW'] - df['SSIM_PCB']
    df['EDGE_DELTA'] = df['EDGE_IOU_RAW'] - df['EDGE_IOU_PCB']
    delta_features = ['PSNR_DELTA', 'SSIM_DELTA', 'EDGE_DELTA']
    
    # Add engineered features
    df['QUALITY_SCORE'] = 0.5*df['PSNR_RAW'] + 0.3*df['SSIM_RAW'] + 0.2*df['EDGE_IOU_RAW']
    df['EDGE_QUALITY'] = df['EDGE_IOU_RAW'] / (df['PSNR_RAW'] / 15 + 0.1)
    df['DEGRADATION'] = (df['PSNR_DELTA'] + df['SSIM_DELTA']*10 + df['EDGE_DELTA']*5) / 3
    df['METRIC_CONSISTENCY'] = np.std([df['PSNR_RAW'], df['SSIM_RAW']*20, df['EDGE_IOU_RAW']*10], axis=0)
    df['SSIM_PSNR_RATIO'] = df['SSIM_RAW'] / (df['PSNR_RAW'] + 1)
    engineered_features = ['QUALITY_SCORE', 'EDGE_QUALITY', 'DEGRADATION', 'METRIC_CONSISTENCY', 'SSIM_PSNR_RATIO']
    
    # Add shuffled-detection features
    shuffled_features_dict = create_shuffled_detection_features(df)
    for feat_name, feat_values in shuffled_features_dict.items():
        df[feat_name] = feat_values
    shuffled_features = list(shuffled_features_dict.keys())
    
    all_features = metric_features + delta_features + engineered_features + shuffled_features
    
    return df[all_features], all_features


def train_improved_model():
    """Train LightGBM with comprehensive feature set optimized for shuffled detection"""
    
    print("=" * 80)
    print("TRAINING IMPROVED MODEL FOR SHUFFLED DETECTION")
    print("=" * 80)
    
    # Load data
    df = load_and_combine_metrics()
    if df is None:
        return
    
    df = create_labels(df)
    X, feature_names = prepare_final_features(df)
    y = df['QUALITY_LABEL']
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Feature breakdown:")
    print(f"    - Metric features: 6")
    print(f"    - Delta features: 3")
    print(f"    - Engineered features: 5")
    print(f"    - Shuffled-detection features: {len(feature_names) - 14}")
    
    print(f"\n  Classes:")
    for cls in sorted(y.unique()):
        count = (y == cls).sum()
        pct = 100 * count / len(y)
        print(f"    {cls:15s}: {count:3d} ({pct:5.1f}%)")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train model
    print(f"\nTraining LightGBM classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=600,
        max_depth=9,
        num_leaves=37,
        learning_rate=0.04,
        random_state=42,
        verbose=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight='balanced',
        min_child_samples=5
    )
    
    model.fit(X_train, y_train, feature_name=feature_names)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nPerformance Metrics:")
    print(f"  Train accuracy: {train_acc:.2%}")
    print(f"  Test accuracy:  {test_acc:.2%}")
    print(f"  Overfitting gap: {(train_acc - test_acc):.2%}")
    
    # Detailed report
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred_test)
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # Variant-specific analysis
    variant_accuracies = {}
    for variant in df['variant'].unique():
        variant_mask = (df.index.isin(X_test.index)) & (df['variant'] == variant)
        if variant_mask.sum() > 0:
            variant_test_indices = df[variant_mask].index
            variant_test_positions = [list(X_test.index).index(i) for i in variant_test_indices if i in X_test.index]
            if variant_test_positions:
                variant_acc = accuracy_score(
                    y_test_labels[variant_test_positions],
                    y_pred_labels[variant_test_positions]
                )
                variant_accuracies[variant] = variant_acc
    
    print(f"\nVariant-Specific Accuracies:")
    for variant in sorted(variant_accuracies.keys()):
        acc = variant_accuracies[variant]
        print(f"  {variant:25s}: {acc:.2%}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:35s}: {row['importance']:8.4f}")
    
    # Save outputs
    model_path = r"d:\Dataset_PCB_Final\results\advice_model\lgb_model_shuffled_optimized.joblib"
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    importance_path = r"d:\Dataset_PCB_Final\results\advice_model\feature_importance_shuffled.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved: {importance_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=sorted(le.classes_))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_, cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Shuffled-Optimized Model', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = r"d:\Dataset_PCB_Final\results\advice_model\confusion_matrix_shuffled.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {cm_path}")
    plt.close()
    
    # Feature importance plot
    plt.figure(figsize=(12, 8))
    top_n = 15
    top_features = feature_importance.head(top_n)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    plt.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=10)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance - Shuffled-Optimized Model', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fi_path = r"d:\Dataset_PCB_Final\results\advice_model\feature_importance_shuffled.png"
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    print(f"✓ Feature importance plot saved: {fi_path}")
    plt.close()
    
    # Summary statistics
    print(f"\n" + "=" * 80)
    print(f"SUMMARY")
    print(f"=" * 80)
    print(f"  Model Type: LightGBM Classifier")
    print(f"  Total Features: {len(feature_names)}")
    print(f"  Training Samples: {len(X_train)}")
    print(f"  Test Samples: {len(X_test)}")
    print(f"  Test Accuracy: {test_acc:.2%}")
    print(f"  Classes: {', '.join(sorted(le.classes_))}")
    print(f"\n  Key Insight:")
    print(f"  - Shuffled detection relies on metric pattern analysis")
    print(f"  - 10 new features designed to capture shuffled characteristics")
    print(f"  - Features capture metric asymmetry, consistency, and degradation patterns")
    

if __name__ == "__main__":
    train_improved_model()
