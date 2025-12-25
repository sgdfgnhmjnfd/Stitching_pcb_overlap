# PCB Stitching & Quality Assessment Pipeline - Quick Start Guide

## Overview

This unified pipeline integrates:
1. **Stitching** - Combines PCB tiles into full images
2. **Evaluation** - Compares stitched vs ground truth
3. **Model Prediction** - Classifies quality and generates advice
4. **Report Generation** - Creates comprehensive analysis

## Quick Usage

### Command Line

```bash
# Run complete pipeline
python.exe scripts/unified_pipeline.py ^
  --tiles_dir "D:\Dataset_PCB_Final\tiles\mach1\mach1_good\1-1\base" ^
  --gt_dir "D:\Dataset_PCB_Final\canonical\mach1_gt\1-1" ^
  --output_dir "D:\Dataset_PCB_Final\results\pipeline_output"
```

### With JSON Output (for programmatic access)

```bash
# Returns JSON result
python.exe scripts/pipeline_cli.py ^
  --tiles "D:\Dataset_PCB_Final\tiles\mach1\mach1_good\1-1\base" ^
  --gt "D:\Dataset_PCB_Final\canonical\mach1_gt\1-1" ^
  --output "D:\Dataset_PCB_Final\results\pipeline_output" ^
  --json
```

### C# Windows Forms Integration

```csharp
var result = PCBPipelineIntegration.RunPCBPipeline(
    @"D:\Dataset_PCB_Final\tiles\mach1\mach1_good\1-1\base",
    @"D:\Dataset_PCB_Final\canonical\mach1_gt\1-1",
    @"D:\Dataset_PCB_Final\results\pipeline_output"
);

if (result.Success)
{
    MessageBox.Show("Pipeline completed!", "Success");
    // Display results
}
else
{
    MessageBox.Show($"Error: {result.Error}", "Failed");
}
```

## Pipeline Steps

### Step 1: Stitching
- Input: Tiles from a specific variant (e.g., mach1_good/1-1/base)
- Process: SIFT feature matching + homography estimation
- Output: `stitched/variant_name_stitched.png`

### Step 2: Evaluation
- Input: Stitched image + Ground truth image
- Process: PSNR, SSIM, Edge IOU computation
- Output: `ALL_METRICS.csv`

### Step 3: Model Prediction
- Input: Metrics data
- Process: LightGBM model classification
- Output: `predictions_with_advice.csv`

### Step 4: Report Generation
- Input: Predictions and metrics
- Process: Generate visualizations and statistics
- Output: Confusion matrices, reports, summaries

## Output Files

### Essential Outputs
- `predictions_with_advice.csv` - All predictions with quality advice
- `confusion_matrix_balanced_*.png` - 4 visualization styles
- `MODEL_REPORT.txt` - Comprehensive analysis report
- `MODEL_METRICS.csv` - Key performance metrics
- `PIPELINE_SUMMARY.json` - Quick summary in JSON

### Optional Outputs
- `stitched/*.png` - Stitched images
- `ALL_METRICS.csv` - Raw evaluation metrics

## Model Performance

**Overall Accuracy:** 99.17% (balanced)

**Per-Class Performance:**
- Fair (medium quality): 99% precision, 97% recall
- Good (high quality): 100% precision, 100% recall
- Poor (low quality): 95% precision, 100% recall

**Quality Thresholds (PSNR_FINAL):**
- Poor: < 12.14 (bottom 25%)
- Fair: 12.14 - 15.36 (middle 50%)
- Good: ≥ 15.36 (top 25%)

## Features Used

1. **PSNR_FINAL** (importance: 1872) - Peak signal-to-noise ratio
2. **IOU_STITCH** (importance: 946) - Intersection over union after stitching
3. **IOU_FINAL** (importance: 828) - Final edge intersection over union
4. **SSIM_FINAL** (importance: 644) - Structural similarity
5. **SSIM_STITCH** (importance: 588) - SSIM after stitching
6. **PSNR_STITCH** (importance: 498) - PSNR after stitching

## Advanced Options

### Skip Steps

```bash
# Skip stitching (use pre-stitched images)
python.exe scripts/unified_pipeline.py ^
  --tiles_dir "..." ^
  --gt_dir "..." ^
  --output_dir "..." ^
  --skip_stitch

# Skip evaluation (use existing metrics)
python.exe scripts/unified_pipeline.py ^
  --tiles_dir "..." ^
  --gt_dir "..." ^
  --output_dir "..." ^
  --skip_eval
```

### Adjust Feature Count

```bash
# Use more SIFT features (slower but more accurate)
python.exe scripts/unified_pipeline.py ^
  --tiles_dir "..." ^
  --gt_dir "..." ^
  --output_dir "..." ^
  --nfeatures 10000

# Use fewer SIFT features (faster)
python.exe scripts/unified_pipeline.py ^
  --tiles_dir "..." ^
  --gt_dir "..." ^
  --output_dir "..." ^
  --nfeatures 2000
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Ensure virtualenv is activated: `.venv\Scripts\activate` |
| "Tiles directory not found" | Check path format: `D:\...\tiles\mach\variant\case\type` |
| "GT directory not found" | Verify canonical directory exists and has GT images |
| Out of memory | Reduce `--nfeatures` or process smaller tile sets |
| Slow stitching | Use lower `--nfeatures` (default 5000) |
| Model prediction failed | Ensure `lgb_model_balanced.joblib` exists in `advice_model/` |

## File Locations

```
D:\Dataset_PCB_Final\
├── scripts/
│   ├── unified_pipeline.py       <- Main pipeline
│   ├── pipeline_cli.py           <- CLI interface
│   ├── stitching/
│   │   └── stitch_tiles.py
│   └── evaluation/
│       ├── evaluation.py
│       ├── run_model_advice.py
│       └── train_balanced_model.py
├── results/
│   ├── advice_model/
│   │   ├── lgb_model_balanced.joblib
│   │   ├── predictions_with_advice.csv
│   │   ├── confusion_matrix_balanced_*.png
│   │   ├── MODEL_REPORT.txt
│   │   └── MODEL_METRICS.csv
│   └── pipeline_output/          <- Pipeline results
│       └── PIPELINE_SUMMARY.json
└── tiles/
    └── [tile directories]
```

## Contact & Support

For detailed information, see:
- `INTEGRATION_GUIDE.md` - C# integration examples
- `results/advice_model/DEPLOYMENT_GUIDE.md` - Model deployment guide
- `results/advice_model/MODEL_REPORT.txt` - Detailed model analysis
