#!/usr/bin/env python3
"""
FULL PIPELINE FINAL: 
1. SMART FIND STITCHED IMAGE
2. FILTER STITCHED IMAGE
3. EVALUATE (CLEAN STITCH vs PRE-CROPPED GT)
"""

import subprocess
from pathlib import Path
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt

PYTHON = sys.executable

# ===============================
# PATH CONFIG
# ===============================

ROOT = Path("D:/Dataset_PCB_Final")

TILES_ROOT   = ROOT / "tiles"
STITCHED_DIR = ROOT / "results" / "stitched"
EVAL_ROOT    = ROOT / "results" / "evaluation"

# Scripts
PCB_FILTER    = ROOT / "scripts" / "stitching" / "pcb_filter.py"
EVAL_SCRIPT   = ROOT / "scripts" / "evaluation" / "evaluation.py"

# GLOBAL BEST GT (Dự phòng)
GLOBAL_GT = ROOT / "ground_truth" / "GT.png"

# ===============================
# UTILS
# ===============================

def run(cmd):
    try:
        return subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        return None

def parse_variant(case_name: str):
    parts = case_name.split('_')
    if len(parts) > 1: return parts[-1]
    return "unknown"

# ===============================
# MAIN
# ===============================

def main():
    if not GLOBAL_GT.exists():
        print(f"[FATAL] Global GT not found: {GLOBAL_GT}")
        return

    STITCHED_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    
    print(f"[INFO] Scanning tiles in: {TILES_ROOT}")
    
    for mach in sorted(TILES_ROOT.iterdir()):
        if not mach.is_dir(): continue
        for cond in sorted(mach.iterdir()):
            if not cond.is_dir(): continue
            for case in sorted(cond.iterdir()):
                if not case.is_dir(): continue
                
                # ===============================
                # TÌM LOCAL GT (ĐÃ CẮT SẴN)
                # ===============================
                possible_gts = [
                    case / "GT.png", 
                    case / "GT.jpg", 
                    case / "gt.png",
                    case / "gt.jpg"
                ]
                
                local_gt_clean = None
                for p in possible_gts:
                    if p.exists():
                        local_gt_clean = p
                        break
                
                if not local_gt_clean:
                    # Nếu không có file GT riêng, dùng Global GT
                    local_gt_clean = GLOBAL_GT
                    print(f"[WARN] Local GT missing for {case.name}. Using Global GT.")

                # Duyệt qua các biến thể
                for variant in sorted(case.iterdir()):
                    if not variant.is_dir(): continue

                    short_name = f"{case.name}_{variant.name}"
                    
                    # Output paths
                    pcb_clean_img = STITCHED_DIR / f"{short_name}_stitched_pcb.png"
                    case_eval_dir = EVAL_ROOT / short_name

                    print(f"\n>>> PROCESSING: {short_name}")

                    # ===============================
                    # 1. TÌM FILE STITCH (PNG/JPG)
                    # ===============================
                    found_stitch = None
                    search_patterns = [
                        f"{short_name}_stitched.png", 
                        f"{short_name}_stitched.jpg",
                        f"*{variant.name}*_stitched.png", 
                        f"*{variant.name}*_stitched.jpg"
                    ]

                    for pat in search_patterns:
                        matches = list(STITCHED_DIR.glob(pat))
                        matches = [m for m in matches if "_pcb" not in m.name and "_filtered" not in m.name and "GT" not in m.name]
                        if matches:
                            found_stitch = max(matches, key=lambda f: f.stat().st_mtime)
                            break
                    
                    if not found_stitch:
                        print(f"   [SKIP] Stitch file missing for {short_name}")
                        continue
                    else:
                        print(f"   [INFO] Input: {found_stitch.name}")
                        print(f"   [INFO] GT: {local_gt_clean.name} (Assumed clean)")

                    # ===============================
                    # 2. FILTER STITCHED IMAGE
                    # ===============================
                    if not pcb_clean_img.exists():
                        print("   [2] Filtering Stitched PCB...")
                        run([PYTHON, PCB_FILTER, found_stitch, "--output", pcb_clean_img])
                    else:
                        print("   [2] Filter Stitched (Exists)")

                    # ===============================
                    # 3. EVALUATION
                    # ===============================
                    # So sánh: 
                    # - Ảnh ghép đã lọc (pcb_clean_img)
                    # - Ảnh GT có sẵn (local_gt_clean) - Đã cắt
                    
                    metrics_csv = case_eval_dir / "metrics.csv"
                    
                    if pcb_clean_img.exists() and local_gt_clean.exists():
                        if not metrics_csv.exists():
                            print(f"   [3] Evaluating...")
                            case_eval_dir.mkdir(parents=True, exist_ok=True)
                            
                            run([
                                PYTHON, EVAL_SCRIPT,
                                "--gt_stitch", local_gt_clean, # Dùng luôn ảnh sạch để so sánh
                                "--gt_final", local_gt_clean,  # Dùng luôn ảnh sạch để chấm điểm
                                "--raw", found_stitch,
                                "--filtered", pcb_clean_img,
                                "--case", short_name,
                                "--out_dir", case_eval_dir
                            ])
                        else:
                            print("   [3] Evaluating (Skipped - Exists)")
                        
                        # Collect Data
                        if metrics_csv.exists():
                            try:
                                with open(metrics_csv, newline="") as f:
                                    reader = csv.DictReader(f)
                                    row = next(reader)
                                row["case"] = short_name
                                row["variant"] = parse_variant(short_name)
                                for k in row:
                                    if k not in ("case", "variant"):
                                        row[k] = float(row[k])
                                all_rows.append(row)
                            except: pass

    # Summary Output
    if all_rows:
        df = pd.DataFrame(all_rows)
        out_csv = EVAL_ROOT / "ALL_METRICS.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[SUCCESS] Saved metrics: {out_csv}")
        
        col_psnr = "PSNR_FINAL" if "PSNR_FINAL" in df.columns else "PSNR_PCB"
        if col_psnr in df.columns:
            print("\nTOP 5 FAIL CASES (Lowest Quality):")
            print(df.sort_values(col_psnr).head(5)[["case", col_psnr]])

if __name__ == "__main__":
    main()