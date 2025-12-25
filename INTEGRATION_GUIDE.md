"""
Integration Guide for C# Windows Forms Application
Shows how to call the Python pipeline from C#
"""

# ============================================================================
# C# INTEGRATION EXAMPLE
# ============================================================================

Example C# code to call the pipeline:

using System;
using System.Diagnostics;
using Newtonsoft.Json;

public class PCBPipelineIntegration
{
    private const string PYTHON_PATH = @"D:\Dataset_PCB_Final\.venv\Scripts\python.exe";
    private const string PIPELINE_SCRIPT = @"d:\Dataset_PCB_Final\scripts\pipeline_cli.py";
    
    public class PipelineResult
    {
        [JsonProperty("success")]
        public bool Success { get; set; }
        
        [JsonProperty("stdout")]
        public string Output { get; set; }
        
        [JsonProperty("stderr")]
        public string Error { get; set; }
        
        [JsonProperty("timestamp")]
        public string Timestamp { get; set; }
    }
    
    public static PipelineResult RunPCBPipeline(
        string tilesDir, 
        string gtDir, 
        string outputDir, 
        int nfeatures = 5000)
    {
        try
        {
            // Build command arguments
            string args = $"--tiles \"{tilesDir}\" " +
                         $"--gt \"{gtDir}\" " +
                         $"--output \"{outputDir}\" " +
                         $"--nfeatures {nfeatures} " +
                         $"--json";
            
            // Create process
            ProcessStartInfo psi = new ProcessStartInfo()
            {
                FileName = PYTHON_PATH,
                Arguments = $"\"{PIPELINE_SCRIPT}\" {args}",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            
            // Run process
            using (Process process = Process.Start(psi))
            {
                string output = process.StandardOutput.ReadToEnd();
                string error = process.StandardError.ReadToEnd();
                process.WaitForExit();
                
                // Parse JSON result
                var result = JsonConvert.DeserializeObject<PipelineResult>(output);
                return result;
            }
        }
        catch (Exception ex)
        {
            return new PipelineResult
            {
                Success = false,
                Error = ex.Message,
                Timestamp = DateTime.Now.ToIso8601String()
            };
        }
    }
    
    // Usage example:
    // 
    // var result = RunPCBPipeline(
    //     @"D:\Dataset_PCB_Final\tiles\mach1\mach1_good\1-1\base",
    //     @"D:\Dataset_PCB_Final\canonical\mach1_gt\1-1",
    //     @"D:\Dataset_PCB_Final\results\pipeline_output"
    // );
    // 
    // if (result.Success)
    // {
    //     MessageBox.Show("Pipeline completed successfully!", "Success");
    //     // Load and display results
    //     LoadPipelineResults(@"D:\Dataset_PCB_Final\results\pipeline_output");
    // }
    // else
    // {
    //     MessageBox.Show($"Pipeline failed: {result.Error}", "Error");
    // }
}

# ============================================================================
# PYTHON PIPELINE INTERFACE
# ============================================================================

Main entry points:

1. UNIFIED PIPELINE (complete workflow):
   python.exe scripts/unified_pipeline.py \
     --tiles_dir "tiles directory" \
     --gt_dir "ground truth directory" \
     --output_dir "output directory"

2. CLI INTERFACE (for C# integration):
   python.exe scripts/pipeline_cli.py \
     --tiles "tiles directory" \
     --gt "ground truth directory" \
     --output "output directory" \
     --json

3. Individual components:
   - Stitching: python.exe scripts/stitching/stitch_tiles.py
   - Evaluation: python.exe scripts/evaluation/evaluation.py
   - Prediction: python.exe scripts/evaluation/run_model_advice.py

# ============================================================================
# EXPECTED OUTPUT FILES
# ============================================================================

After pipeline completion:

1. Stitched Images:
   results/pipeline_output/stitched/*.png

2. Evaluation Metrics:
   results/evaluation/ALL_METRICS.csv

3. Model Predictions:
   results/advice_model/predictions_with_advice.csv

4. Visualizations:
   results/advice_model/confusion_matrix_balanced_*.png

5. Reports:
   results/advice_model/MODEL_REPORT.txt
   results/advice_model/MODEL_METRICS.csv
   results/pipeline_output/PIPELINE_SUMMARY.json

# ============================================================================
# RETURN JSON STRUCTURE
# ============================================================================

{
  "success": true,
  "stdout": "...pipeline output...",
  "stderr": "",
  "timestamp": "2025-12-25T15:00:00"
}

# ============================================================================
# ERROR HANDLING
# ============================================================================

Common errors and solutions:

1. "Python not found"
   - Ensure .venv is activated
   - Check Python path: D:\Dataset_PCB_Final\.venv\Scripts\python.exe

2. "Tiles directory not found"
   - Verify tiles path is correct
   - Format: D:\Dataset_PCB_Final\tiles\mach1\mach1_good\1-1\base

3. "GT directory not found"
   - Verify ground truth path exists
   - Format: D:\Dataset_PCB_Final\canonical\mach1_gt\1-1

4. Out of memory during stitching
   - Reduce nfeatures (default: 5000)
   - Example: --nfeatures 2000

5. Model prediction failed
   - Ensure ALL_METRICS.csv exists
   - Check model file: results/advice_model/lgb_model_balanced.joblib
