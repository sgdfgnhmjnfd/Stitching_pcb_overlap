#!/usr/bin/env python3
"""
PLOT METRICS V2 (Dual GT Support)
- Visualizes Stitch Accuracy (vs Local GT)
- Visualizes Final Quality (vs Global Best GT)
- Adds data labels (notes) directly on charts
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path("D:/Dataset_PCB_Final/results/evaluation")
OUT_DIR = ROOT / "_plots_v2"

# ===============================
# 1. READ DATA
# ===============================

def read_metrics(root_dir):
    data = {
        "cases": [],
        "psnr_stitch": [], "psnr_final": [],
        "ssim_stitch": [], "ssim_final": [],
        "iou_stitch": [],  "iou_final": []
    }

    # Quét tất cả folder con
    for case_dir in sorted(root_dir.iterdir()):
        if not case_dir.is_dir(): continue
        
        csv_path = case_dir / "metrics.csv"
        if not csv_path.exists(): continue

        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                row = next(reader) # Lấy dòng đầu tiên
                
                # Bỏ qua nếu CSV chưa update format mới
                if "PSNR_STITCH" not in row: continue

                data["cases"].append(row["case"])
                data["psnr_stitch"].append(float(row["PSNR_STITCH"]))
                data["psnr_final"].append(float(row["PSNR_FINAL"]))
                data["ssim_stitch"].append(float(row["SSIM_STITCH"]))
                data["ssim_final"].append(float(row["SSIM_FINAL"]))
                data["iou_stitch"].append(float(row["IOU_STITCH"]))
                data["iou_final"].append(float(row["IOU_FINAL"]))
        except Exception as e:
            print(f"[WARN] Error reading {csv_path}: {e}")

    return data

# ===============================
# 2. PLOTTING WITH NOTES
# ===============================

def plot_bar_chart(cases, val1, val2, label1, label2, title, ylabel, filename):
    if not cases: return

    x = np.arange(len(cases))
    width = 0.35  # Độ rộng cột

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Vẽ 2 cột song song
    rects1 = ax.bar(x - width/2, val1, width, label=label1, color='royalblue', alpha=0.8)
    rects2 = ax.bar(x + width/2, val2, width, label=label2, color='darkorange', alpha=0.8)

    # Label trục
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=45, ha='right', fontsize=10)
    ax.legend()

    # --- ADD DATA LABELS (NOTES) ---
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)

    autolabel(rects1)
    autolabel(rects2)

    # Grid mờ phía sau
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=150)
    print(f"[SAVED] {out_path}")
    plt.close()

# ===============================
# MAIN
# ===============================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(">>> Reading Metrics...")
    
    data = read_metrics(ROOT)
    count = len(data["cases"])
    
    if count == 0:
        print("[ERR] No valid metrics found (Check CSV format).")
        return

    print(f"> Found {count} cases. Generating plots...")

    # 1. PSNR Plot
    plot_bar_chart(
        data["cases"], 
        data["psnr_stitch"], data["psnr_final"],
        "Stitch Accuracy (vs Local GT)", 
        "Final Quality (vs Best GT)",
        "PSNR Comparison (Higher is Better)",
        "PSNR (dB)",
        "chart_psnr.png"
    )

    # 2. SSIM Plot
    plot_bar_chart(
        data["cases"], 
        data["ssim_stitch"], data["ssim_final"],
        "Stitch Structure", 
        "Final Structure",
        "SSIM Comparison (Max 1.0)",
        "SSIM Index",
        "chart_ssim.png"
    )

    # 3. IoU Plot
    plot_bar_chart(
        data["cases"], 
        data["iou_stitch"], data["iou_final"],
        "Edge Match (Stitch)", 
        "Edge Match (Final)",
        "Edge IoU Comparison (Shape Accuracy)",
        "IoU Score",
        "chart_iou.png"
    )

    print("\n===== PLOTTING DONE =====")
    print(f"Open folder: {OUT_DIR}")

if __name__ == "__main__":
    main()