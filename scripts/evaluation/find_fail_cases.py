#!/usr/bin/env python3
"""
FIND WORST CASE UTILITY
- Scans all evaluation results
- Finds the case with lowest SSIM (Worst quality)
- Generates a zoomed-in error map for inspection
"""

from pathlib import Path
import csv
import cv2
import numpy as np

# Đường dẫn gốc (Cần khớp với config trong C#)
ROOT = Path("D:/Dataset_PCB_Final/results/evaluation")

def main():
    if not ROOT.exists():
        print(f"[ERR] Directory not found: {ROOT}")
        return

    records = []
    print(f"[INFO] Scanning results in: {ROOT}")

    # 1. Duyệt qua các folder kết quả
    for case_dir in ROOT.iterdir():
        if not case_dir.is_dir(): continue

        csv_path = case_dir / "metrics.csv"
        if not csv_path.exists(): continue

        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                row = next(reader)
                
                # Logic tương thích ngược (Hỗ trợ cả format Cũ và Mới)
                if "SSIM_FINAL" in row:
                    score = float(row["SSIM_FINAL"]) # Format Mới (Dual GT)
                elif "SSIM_PCB" in row:
                    score = float(row["SSIM_PCB"])   # Format Cũ
                else:
                    continue # Không tìm thấy cột điểm số
                
                records.append({
                    "name": case_dir.name,
                    "ssim": score,
                    "dir": case_dir
                })
        except Exception as e:
            print(f"[WARN] Skipped {case_dir.name}: {e}")

    if not records:
        print("[WARN] No valid evaluation records found.")
        return

    # 2. Tìm ca tệ nhất (SSIM thấp nhất)
    # Sắp xếp tăng dần theo SSIM -> Phần tử đầu tiên là tệ nhất
    records.sort(key=lambda x: x["ssim"])
    worst = records[0]

    print(f"\n" + "="*40)
    print(f" WORST CASE FOUND: {worst['name']}")
    print(f" SSIM Score: {worst['ssim']:.4f}")
    print(f"="*40)

    # 3. Tạo ảnh Zoom Error (Cắt vùng giữa Heatmap)
    # Ưu tiên tìm tên file mới, nếu không có thì tìm tên cũ
    heatmap_candidates = [
        "heatmap_final.png", # Tên mới (Dual GT)
        "heatmap_diff.png",  # Tên cũ 1
        "heatmap_roi.png"    # Tên cũ 2
    ]
    
    heat_path = None
    for fname in heatmap_candidates:
        p = worst["dir"] / fname
        if p.exists():
            heat_path = p
            break
    
    if heat_path:
        print(f"[PROCESS] Generating zoom error from: {heat_path.name}")
        heat = cv2.imread(str(heat_path))
        
        if heat is not None:
            h, w = heat.shape[:2]
            
            # Cắt vùng trung tâm 50% (Thường lỗi hay nằm ở giữa hoặc lệch pha tổng thể)
            cy, cx = h // 2, w // 2
            dy, dx = h // 4, w // 4
            
            # Crop vùng [1/4 -> 3/4]
            crop = heat[cy-dy : cy+dy, cx-dx : cx+dx]
            
            # Resize phóng to lên 2 lần cho dễ nhìn
            crop_zoom = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

            # Vẽ khung đỏ bao quanh để ngầu
            h_z, w_z = crop_zoom.shape[:2]
            cv2.rectangle(crop_zoom, (0,0), (w_z-1, h_z-1), (0,0,255), 4)

            out_file = worst["dir"] / "zoom_error_analysis.png"
            cv2.imwrite(str(out_file), crop_zoom)
            
            print(f"  -> Saved: {out_file}")
            
            # Mở file lên xem luôn (Windows only)
            import os
            os.startfile(str(out_file))
        else:
            print("[ERR] Could not read heatmap image.")
    else:
        print("[WARN] Heatmap file not found for this case.")

if __name__ == "__main__":
    main()