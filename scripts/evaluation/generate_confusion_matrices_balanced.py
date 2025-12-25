"""
Create comprehensive confusion matrices with balanced model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_predict():
    """Load data, model, and make predictions"""
    # Load data
    metrics_file = Path(r"d:\Dataset_PCB_Final\results\evaluation\ALL_METRICS.csv")
    df = pd.read_csv(metrics_file)
    
    # Load model and scaler
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
    
    # Prepare features
    features = ['PSNR_STITCH', 'SSIM_STITCH', 'IOU_STITCH', 'PSNR_FINAL', 'SSIM_FINAL', 'IOU_FINAL']
    X = df[features].copy()
    X_scaled = scaler.transform(X)
    
    # Predict
    y_pred_encoded = model.predict(X_scaled)
    y_true_encoded = pd.factorize(df['label'])[0]
    
    # Decode to labels
    class_labels = ['fair', 'good', 'poor']
    y_true = [class_labels[i] for i in y_true_encoded]
    y_pred = [class_labels[i] for i in y_pred_encoded]
    
    return df, y_true, y_pred, model, class_labels

def plot_confusion_matrix_1():
    """Style 1: Classic heatmap with counts"""
    df, y_true, y_pred, model, class_labels = load_and_predict()
    
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'}, ax=ax, linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'weight': 'bold'})
    
    ax.set_title('Confusion Matrix: PCB Stitching Quality Classification\n(Classic Heatmap)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\confusion_matrix_balanced_01.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_confusion_matrix_2():
    """Style 2: Normalized percentages"""
    df, y_true, y_pred, model, class_labels = load_and_predict()
    
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='YlOrRd',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Percentage'}, ax=ax, linewidths=2, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_title('Confusion Matrix: PCB Stitching Quality Classification\n(Normalized Percentages)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\confusion_matrix_balanced_02.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_confusion_matrix_3():
    """Style 3: Dual display (counts + percentages)"""
    df, y_true, y_pred, model, class_labels = load_and_predict()
    
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotations with both counts and percentages
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"
    
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='RdYlGn_r',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'}, ax=ax, linewidths=2, linecolor='white',
                annot_kws={'size': 11, 'weight': 'bold'})
    
    ax.set_title('Confusion Matrix: PCB Stitching Quality Classification\n(Count + Percentage)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\confusion_matrix_balanced_03.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_confusion_matrix_4():
    """Style 4: With metrics on the side"""
    df, y_true, y_pred, model, class_labels = load_and_predict()
    
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.6], wspace=0.3)
    
    # Confusion matrix
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'}, ax=ax1, linewidths=2, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    # Metrics
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    
    metrics_text = f"""
PERFORMANCE METRICS
{'='*30}

Overall Accuracy: {accuracy:.1%}

Per-Class Metrics:
{'-'*30}
"""
    
    for class_name in class_labels:
        metrics = report[class_name]
        metrics_text += f"\n{class_name.upper()}:\n"
        metrics_text += f"  Precision: {metrics['precision']:.1%}\n"
        metrics_text += f"  Recall: {metrics['recall']:.1%}\n"
        metrics_text += f"  F1-Score: {metrics['f1-score']:.1%}\n"
    
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('PCB Stitching Quality Classification - Confusion Matrix with Metrics', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = Path(r"d:\Dataset_PCB_Final\results\advice_model\confusion_matrix_balanced_04.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def print_statistics():
    """Print detailed statistics"""
    df, y_true, y_pred, model, class_labels = load_and_predict()
    
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*80)
    
    print(f"\nConfusion Matrix:")
    print(f"{'':15} " + " ".join(f"{label:>12}" for label in class_labels))
    for i, label in enumerate(class_labels):
        print(f"{label:>15} " + " ".join(f"{cm[i, j]:>12}" for j in range(len(class_labels))))
    
    print(f"\n\nOverall Accuracy: {accuracy:.2%}")
    
    print(f"\n\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

def main():
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRICES (BALANCED MODEL)")
    print("="*80)
    
    print_statistics()
    
    print("\n[GENERATING VISUALIZATIONS...]")
    plot_confusion_matrix_1()
    plot_confusion_matrix_2()
    plot_confusion_matrix_3()
    plot_confusion_matrix_4()
    
    print("\n" + "="*80)
    print("✓ ALL CONFUSION MATRICES GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("  1. confusion_matrix_balanced_01.png - Classic heatmap style")
    print("  2. confusion_matrix_balanced_02.png - Normalized percentages")
    print("  3. confusion_matrix_balanced_03.png - Count + Percentage combined")
    print("  4. confusion_matrix_balanced_04.png - Matrix with metrics panel")

if __name__ == "__main__":
    main()
