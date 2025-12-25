"""
Unified PCB Stitching & Quality Assessment Pipeline
Combines: Stitching -> Evaluation -> Model Prediction -> Advice
"""

import argparse
import sys
from pathlib import Path
import subprocess
import json
import pandas as pd
from datetime import datetime

def print_header(title):
    """Print section header"""
    print("\n" + "="*100)
    print(f"  {title}")
    print("="*100)

def run_stitching(tiles_dir, output_dir, nfeatures=5000):
    """Run tile stitching"""
    print_header("STEP 1: STITCHING TILES")
    
    script = Path(r"d:\Dataset_PCB_Final\scripts\stitching\stitch_tiles.py")
    cmd = [
        "python.exe",
        str(script),
        "--tiles_dir", str(tiles_dir),
        "--out_root", str(output_dir),
        "--mode", "stitch",
        "--nfeatures", str(nfeatures)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR in stitching: {result.stderr}")
        return False
    
    print(result.stdout)
    print("✓ Stitching completed")
    return True

def run_evaluation(stitched_dir, gt_dir, output_dir):
    """Run evaluation"""
    print_header("STEP 2: EVALUATING STITCHED IMAGES")
    
    script = Path(r"d:\Dataset_PCB_Final\scripts\evaluation\evaluation.py")
    cmd = [
        "python.exe",
        str(script),
        "--stitch", str(stitched_dir),
        "--gt", str(gt_dir),
        "--out", str(output_dir)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR in evaluation: {result.stderr}")
        return False
    
    print(result.stdout)
    print("✓ Evaluation completed")
    return True

def run_model_prediction(metrics_file, output_dir):
    """Run model prediction and generate advice"""
    print_header("STEP 3: MODEL PREDICTION & ADVICE GENERATION")
    
    script = Path(r"d:\Dataset_PCB_Final\scripts\evaluation\run_model_advice.py")
    
    # For now, just run the existing script which uses ALL_METRICS.csv
    cmd = [
        "python.exe",
        str(script)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR in model prediction: {result.stderr}")
        return False
    
    print(result.stdout)
    print("✓ Model prediction completed")
    return True

def generate_summary_report(output_dir):
    """Generate summary report"""
    print_header("STEP 4: GENERATING SUMMARY REPORT")
    
    advice_dir = Path(r"d:\Dataset_PCB_Final\results\advice_model")
    predictions_file = advice_dir / "predictions_with_advice.csv"
    metrics_file = advice_dir / "MODEL_METRICS.csv"
    
    if not predictions_file.exists() or not metrics_file.exists():
        print("Warning: Could not find prediction or metrics files")
        return False
    
    # Load data
    predictions = pd.read_csv(predictions_file)
    metrics = pd.read_csv(metrics_file)
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(predictions),
        'prediction_distribution': predictions['prediction'].value_counts().to_dict(),
        'model_accuracy': float(metrics[metrics['Metric'] == 'Overall Accuracy']['Value'].values[0]),
        'good_samples': len(predictions[predictions['prediction'] == 'good']),
        'fair_samples': len(predictions[predictions['prediction'] == 'fair']),
        'poor_samples': len(predictions[predictions['prediction'] == 'poor']),
    }
    
    # Save summary
    summary_path = output_dir / "PIPELINE_SUMMARY.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'─'*100}")
    print(f"Total Samples Analyzed: {summary['total_samples']}")
    print(f"  ✓ Good Quality: {summary['good_samples']} samples")
    print(f"  ~ Fair Quality: {summary['fair_samples']} samples")
    print(f"  ✗ Poor Quality: {summary['poor_samples']} samples")
    print(f"\nModel Accuracy: {summary['model_accuracy']:.2%}")
    print(f"Summary saved: {summary_path}")
    print(f"{'─'*100}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="PCB Stitching & Quality Assessment Pipeline")
    parser.add_argument("--tiles_dir", type=str, required=True, help="Input tiles directory")
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground truth directory")
    parser.add_argument("--output_dir", type=str, default=r"d:\Dataset_PCB_Final\results\pipeline_output", 
                       help="Output directory")
    parser.add_argument("--nfeatures", type=int, default=5000, help="Number of SIFT features")
    parser.add_argument("--skip_stitch", action='store_true', help="Skip stitching step")
    parser.add_argument("--skip_eval", action='store_true', help="Skip evaluation step")
    
    args = parser.parse_args()
    
    # Validate inputs
    tiles_dir = Path(args.tiles_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    
    if not tiles_dir.exists():
        print(f"ERROR: Tiles directory not found: {tiles_dir}")
        return False
    
    if not gt_dir.exists():
        print(f"ERROR: GT directory not found: {gt_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*100)
    print("  PCB STITCHING & QUALITY ASSESSMENT PIPELINE")
    print("="*100)
    print(f"\nInput Tiles: {tiles_dir}")
    print(f"Ground Truth: {gt_dir}")
    print(f"Output Dir: {output_dir}")
    
    # Run pipeline steps
    try:
        if not args.skip_stitch:
            if not run_stitching(tiles_dir, output_dir, args.nfeatures):
                return False
        
        if not args.skip_eval:
            stitched_dir = output_dir / "stitched"
            if not run_evaluation(stitched_dir, gt_dir, output_dir):
                return False
        
        # Always run model prediction
        if not run_model_prediction(None, output_dir):
            return False
        
        # Generate summary
        if not generate_summary_report(output_dir):
            return False
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"\n✓ All steps completed successfully!")
        print(f"✓ Results saved to: {output_dir}")
        print("\nGenerated files:")
        print(f"  - predictions_with_advice.csv (in advice_model)")
        print(f"  - confusion_matrix_balanced_*.png (in advice_model)")
        print(f"  - MODEL_REPORT.txt (in advice_model)")
        print(f"  - MODEL_METRICS.csv (in advice_model)")
        print(f"  - PIPELINE_SUMMARY.json (in output_dir)")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
