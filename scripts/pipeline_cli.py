"""
Simple CLI Interface for PCB Stitching Pipeline
Can be called from C# Windows Forms application or command line
"""

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import sys

def run_pipeline(tiles_dir, gt_dir, output_dir, nfeatures=5000):
    """Run complete pipeline and return results"""
    
    script = Path(r"d:\Dataset_PCB_Final\scripts\unified_pipeline.py")
    
    cmd = [
        "python.exe",
        str(script),
        "--tiles_dir", str(tiles_dir),
        "--gt_dir", str(gt_dir),
        "--output_dir", str(output_dir),
        "--nfeatures", str(nfeatures)
    ]
    
    print(f"Starting pipeline...")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    output = {
        'success': result.returncode == 0,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'timestamp': datetime.now().isoformat()
    }
    
    return output

def main():
    parser = argparse.ArgumentParser(description="PCB Pipeline CLI Interface")
    parser.add_argument("--tiles", type=str, required=True, help="Tiles directory")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth directory")
    parser.add_argument("--output", type=str, default=r"d:\Dataset_PCB_Final\results\pipeline_output",
                       help="Output directory")
    parser.add_argument("--nfeatures", type=int, default=5000, help="Number of features")
    parser.add_argument("--json", action='store_true', help="Output as JSON")
    
    args = parser.parse_args()
    
    result = run_pipeline(args.tiles, args.gt, args.output, args.nfeatures)
    
    if args.json:
        # Output JSON for programmatic access
        print(json.dumps(result, indent=2))
    else:
        # Output human-readable
        print("="*100)
        print("PIPELINE EXECUTION RESULT")
        print("="*100)
        print(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"Timestamp: {result['timestamp']}")
        print("\nOutput:")
        print(result['stdout'])
        if result['stderr']:
            print("\nErrors:")
            print(result['stderr'])
        print("="*100)
    
    return 0 if result['success'] else 1

if __name__ == "__main__":
    sys.exit(main())
