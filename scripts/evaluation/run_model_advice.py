"""
Run inference on model with detailed analysis and recommendations
Shows per-sample predictions, confidence, and detailed advice
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_model_and_scaler():
    """Load trained model and scaler"""
    model_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\lgb_model_balanced.joblib")
    scaler_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\feature_scaler_balanced.joblib")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"✓ Model loaded from {model_path}")
    print(f"✓ Scaler loaded from {scaler_path}")
    print(f"✓ Classes: {model.classes_}")
    
    return model, scaler

def generate_advice(row, prediction, probabilities, class_names):
    """Generate detailed advice based on prediction and probabilities"""
    case = row['case']
    variant = row['variant']
    psnr_final = row['PSNR_FINAL']
    ssim_final = row['SSIM_FINAL']
    iou_final = row['IOU_FINAL']
    
    # Map class IDs to labels
    class_labels = ['fair', 'good', 'poor']
    pred_label = class_labels[prediction]
    
    advice = {
        'case': case,
        'variant': variant,
        'prediction': pred_label,
        'confidence': max(probabilities),
        'psnr_final': psnr_final,
        'ssim_final': ssim_final,
        'iou_final': iou_final,
    }
    
    # Generate text advice
    if pred_label == 'good':
        advice['text'] = f"✓ EXCELLENT stitch quality. PSNR={psnr_final:.2f}, SSIM={ssim_final:.4f}. Ready for production."
    elif pred_label == 'fair':
        advice['text'] = f"~ ACCEPTABLE but can be improved. PSNR={psnr_final:.2f}, SSIM={ssim_final:.4f}. Review stitching parameters."
        if psnr_final < 16:
            advice['text'] += " Focus on alignment accuracy."
        if ssim_final < 0.60:
            advice['text'] += " Consider noise reduction."
    else:  # poor
        advice['text'] = f"✗ POOR stitch quality. PSNR={psnr_final:.2f}, SSIM={ssim_final:.4f}. Requires re-stitching."
        issues = []
        if psnr_final < 12:
            issues.append("very low PSNR - misalignment")
        if ssim_final < 0.54:
            issues.append("low SSIM - structural difference")
        if iou_final < 0.35:
            issues.append("poor edge IOU")
        advice['text'] += " Issues: " + ", ".join(issues) if issues else ""
    
    # Add probabilities for all classes
    for i, class_name in enumerate(class_labels):
        advice[f'prob_{class_name}'] = probabilities[i]
    
    return advice

def run_inference():
    """Run full inference pipeline"""
    print("\n" + "="*100)
    print("RUNNING MODEL INFERENCE WITH DETAILED ADVICE")
    print("="*100)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    if model is None:
        return
    
    # Load data
    metrics_file = Path(r"d:\Dataset_PCB_Final\results\evaluation\ALL_METRICS.csv")
    df = pd.read_csv(metrics_file)
    print(f"\n✓ Loaded {len(df)} samples from {metrics_file}")
    
    # Prepare features
    features = ['PSNR_STITCH', 'SSIM_STITCH', 'IOU_STITCH', 'PSNR_FINAL', 'SSIM_FINAL', 'IOU_FINAL']
    X = df[features].copy()
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    
    print(f"\n✓ Made {len(y_pred)} predictions")
    
    # Generate advice for each sample
    all_advice = []
    for idx, (i, row) in enumerate(df.iterrows()):
        pred = model.classes_[y_pred[idx]]
        proba = y_proba[idx]
        advice = generate_advice(row, pred, proba, model.classes_)
        all_advice.append(advice)
    
    advice_df = pd.DataFrame(all_advice)
    
    # Save results
    output_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\predictions_with_advice.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    advice_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved detailed predictions to {output_path}")
    
    # Print statistics
    print("\n" + "="*100)
    print("PREDICTION STATISTICS")
    print("="*100)
    print(f"\nDistribution of predictions:")
    print(advice_df['prediction'].value_counts())
    
    print(f"\nAverage confidence by class:")
    class_labels = ['fair', 'good', 'poor']
    for i, class_name in enumerate(model.classes_):
        conf = advice_df[advice_df['prediction'] == class_name]['confidence'].mean()
        print(f"  {class_labels[i]}: {conf:.2%}")
    
    print(f"\nMetrics by prediction:")
    for i, class_name in enumerate(model.classes_):
        subset = advice_df[advice_df['prediction'] == class_name]
        class_label = ['fair', 'good', 'poor'][i]
        print(f"\n  {class_label.upper()} ({len(subset)} samples):")
        print(f"    Avg PSNR: {subset['psnr_final'].mean():.2f}")
        print(f"    Avg SSIM: {subset['ssim_final'].mean():.4f}")
        print(f"    Avg IOU:  {subset['iou_final'].mean():.4f}")
    
    # Print sample advice
    print("\n" + "="*100)
    print("SAMPLE ADVICE (first 10 samples)")
    print("="*100)
    for idx in range(min(10, len(advice_df))):
        row = advice_df.iloc[idx]
        print(f"\n[{idx+1}] {row['case']} ({row['variant']})")
        print(f"    Prediction: {row['prediction'].upper()} (confidence: {row['confidence']:.1%})")
        print(f"    Metrics: PSNR={row['psnr_final']:.2f}, SSIM={row['ssim_final']:.4f}, IOU={row['iou_final']:.4f}")
        print(f"    Advice: {row['text']}")
    
    print("\n" + "="*100)
    print(f"✓ COMPLETED - Full results in {output_path}")
    print("="*100)

if __name__ == "__main__":
    run_inference()
