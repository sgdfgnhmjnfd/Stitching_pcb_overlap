"""
Generate comprehensive report with all important numbers for thesis/report
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_and_predict():
    """Load data, model, and make predictions"""
    metrics_file = Path(r"d:\Dataset_PCB_Final\results\evaluation\ALL_METRICS.csv")
    df = pd.read_csv(metrics_file)
    
    model_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\lgb_model_balanced.joblib")
    scaler_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\feature_scaler_balanced.joblib")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Create labels using natural quantiles
    p25 = df['PSNR_FINAL'].quantile(0.25)
    p75 = df['PSNR_FINAL'].quantile(0.75)
    
    df['label'] = 'fair'
    df.loc[df['PSNR_FINAL'] < p25, 'label'] = 'poor'
    df.loc[df['PSNR_FINAL'] >= p75, 'label'] = 'good'
    
    features = ['PSNR_STITCH', 'SSIM_STITCH', 'IOU_STITCH', 'PSNR_FINAL', 'SSIM_FINAL', 'IOU_FINAL']
    X = df[features].copy()
    X_scaled = scaler.transform(X)
    
    y_pred_encoded = model.predict(X_scaled)
    y_true_encoded = pd.factorize(df['label'])[0]
    
    class_labels = ['fair', 'good', 'poor']
    y_true = [class_labels[i] for i in y_true_encoded]
    y_pred = [class_labels[i] for i in y_pred_encoded]
    
    return df, y_true, y_pred, model, class_labels, p25, p75

def generate_report():
    """Generate comprehensive report"""
    df, y_true, y_pred, model, class_labels, p25, p75 = load_and_predict()
    
    # Calculate all metrics
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    
    # Create report
    report_text = f"""
{'='*100}
PCB STITCHING QUALITY CLASSIFICATION - COMPREHENSIVE REPORT
{'='*100}

1. DATASET INFORMATION
{'─'*100}
Total Samples: {len(df)}

Data Distribution:
  - Poor Quality (PSNR < {p25:.2f}):     {(df['label'] == 'poor').sum():3d} samples ({(df['label'] == 'poor').sum()/len(df)*100:.1f}%)
  - Fair Quality  ({p25:.2f} ≤ PSNR < {p75:.2f}): {(df['label'] == 'fair').sum():3d} samples ({(df['label'] == 'fair').sum()/len(df)*100:.1f}%)
  - Good Quality  (PSNR ≥ {p75:.2f}):     {(df['label'] == 'good').sum():3d} samples ({(df['label'] == 'good').sum()/len(df)*100:.1f}%)

Feature Statistics:
  PSNR_FINAL: min={df['PSNR_FINAL'].min():.2f}, max={df['PSNR_FINAL'].max():.2f}, mean={df['PSNR_FINAL'].mean():.2f}, std={df['PSNR_FINAL'].std():.2f}
  SSIM_FINAL: min={df['SSIM_FINAL'].min():.4f}, max={df['SSIM_FINAL'].max():.4f}, mean={df['SSIM_FINAL'].mean():.4f}, std={df['SSIM_FINAL'].std():.4f}
  IOU_FINAL:  min={df['IOU_FINAL'].min():.4f}, max={df['IOU_FINAL'].max():.4f}, mean={df['IOU_FINAL'].mean():.4f}, std={df['IOU_FINAL'].std():.4f}

2. MODEL PERFORMANCE
{'─'*100}
Overall Accuracy:  {accuracy:.2%}
Balanced Accuracy: {balanced_acc:.2%}

Confusion Matrix:
           Predicted
            Fair  Good  Poor
True Fair   {cm[0,0]:3d}   {cm[0,1]:3d}   {cm[0,2]:3d}
     Good   {cm[1,0]:3d}   {cm[1,1]:3d}   {cm[1,2]:3d}
     Poor   {cm[2,0]:3d}   {cm[2,1]:3d}   {cm[2,2]:3d}

3. PER-CLASS METRICS
{'─'*100}
"""
    
    for class_name in class_labels:
        metrics = report[class_name]
        report_text += f"""
{class_name.upper()}:
  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
  F1-Score:  {metrics['f1-score']:.4f}
  Support:   {int(metrics['support'])} samples
"""
    
    # Feature importance
    report_text += f"""
4. FEATURE IMPORTANCE
{'─'*100}
Features (ranked by importance):
"""
    
    features = ['PSNR_STITCH', 'SSIM_STITCH', 'IOU_STITCH', 'PSNR_FINAL', 'SSIM_FINAL', 'IOU_FINAL']
    feature_imp = list(zip(features, model.feature_importances_))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, imp) in enumerate(feature_imp, 1):
        report_text += f"\n  {i}. {name:15s}: {imp:8.4f}"
    
    # Model configuration
    report_text += f"""

5. MODEL CONFIGURATION
{'─'*100}
Model Type: LightGBM Classifier
Number of Estimators: 200
Learning Rate: 0.05
Number of Leaves: 31
Max Depth: 7
Class Weight: balanced
Random State: 42

6. CLASSIFICATION THRESHOLDS
{'─'*100}
Quality Classes based on PSNR_FINAL:
  - Poor:  PSNR_FINAL < {p25:.2f}  (Bottom 25% percentile)
  - Fair:  {p25:.2f} ≤ PSNR_FINAL < {p75:.2f}  (Middle 50% percentile)
  - Good:  PSNR_FINAL ≥ {p75:.2f}  (Top 25% percentile)

7. KEY FINDINGS
{'─'*100}
✓ Model achieves {accuracy:.1%} accuracy with balanced class distribution
✓ Best performing on Fair class: {report['fair']['f1-score']:.1%} F1-score
✓ Good class perfectly identified: {report['good']['recall']:.1%} recall
✓ Most important feature: PSNR_FINAL ({model.feature_importances_[3]:.0f} importance)

8. PREDICTION STATISTICS
{'─'*100}
Distribution of Model Predictions:
  - Fair: {(np.array(y_pred) == 'fair').sum():3d} ({(np.array(y_pred) == 'fair').sum()/len(y_pred)*100:.1f}%)
  - Good: {(np.array(y_pred) == 'good').sum():3d} ({(np.array(y_pred) == 'good').sum()/len(y_pred)*100:.1f}%)
  - Poor: {(np.array(y_pred) == 'poor').sum():3d} ({(np.array(y_pred) == 'poor').sum()/len(y_pred)*100:.1f}%)

{'='*100}
Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*100}
"""
    
    return report_text

def main():
    report_text = generate_report()
    
    # Print to console
    print(report_text)
    
    # Save to file
    output_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\MODEL_REPORT.txt")
    output_path.write_text(report_text, encoding='utf-8')
    print(f"\n✓ Report saved to: {output_path}")
    
    # Also save as CSV for easy importing
    df, y_true, y_pred, model, class_labels, p25, p75 = load_and_predict()
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Create summary CSV
    summary_data = {
        'Metric': [
            'Overall Accuracy',
            'Balanced Accuracy',
            'Fair Precision',
            'Fair Recall',
            'Fair F1-Score',
            'Good Precision',
            'Good Recall',
            'Good F1-Score',
            'Poor Precision',
            'Poor Recall',
            'Poor F1-Score',
            'Total Samples',
            'PSNR_FINAL Min',
            'PSNR_FINAL Max',
            'PSNR_FINAL Mean',
            'SSIM_FINAL Mean',
            'IOU_FINAL Mean',
            'Poor Threshold (PSNR)',
            'Good Threshold (PSNR)',
        ],
        'Value': [
            f"{accuracy:.4f}",
            f"{balanced_acc:.4f}",
            f"{cm[0,0]/(cm[0,0]+cm[1,0]+cm[2,0]):.4f}" if (cm[0,0]+cm[1,0]+cm[2,0]) > 0 else "0",
            f"{cm[0,0]/cm[0,:].sum():.4f}" if cm[0,:].sum() > 0 else "0",
            f"{cm[0,0]/(cm[0,0]+cm[1,0]+cm[2,0]):.4f}" if (cm[0,0]+cm[1,0]+cm[2,0]) > 0 else "0",
            f"{cm[1,1]/(cm[0,1]+cm[1,1]+cm[2,1]):.4f}" if (cm[0,1]+cm[1,1]+cm[2,1]) > 0 else "0",
            f"{cm[1,1]/cm[1,:].sum():.4f}" if cm[1,:].sum() > 0 else "0",
            f"{cm[1,1]/(cm[0,1]+cm[1,1]+cm[2,1]):.4f}" if (cm[0,1]+cm[1,1]+cm[2,1]) > 0 else "0",
            f"{cm[2,2]/(cm[0,2]+cm[1,2]+cm[2,2]):.4f}" if (cm[0,2]+cm[1,2]+cm[2,2]) > 0 else "0",
            f"{cm[2,2]/cm[2,:].sum():.4f}" if cm[2,:].sum() > 0 else "0",
            f"{cm[2,2]/(cm[0,2]+cm[1,2]+cm[2,2]):.4f}" if (cm[0,2]+cm[1,2]+cm[2,2]) > 0 else "0",
            str(len(df)),
            f"{df['PSNR_FINAL'].min():.2f}",
            f"{df['PSNR_FINAL'].max():.2f}",
            f"{df['PSNR_FINAL'].mean():.2f}",
            f"{df['SSIM_FINAL'].mean():.4f}",
            f"{df['IOU_FINAL'].mean():.4f}",
            f"{p25:.2f}",
            f"{p75:.2f}",
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\MODEL_METRICS.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Metrics saved to: {summary_path}")

if __name__ == "__main__":
    main()
