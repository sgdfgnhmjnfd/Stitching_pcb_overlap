#!/usr/bin/env python3
"""
PCB Filter V5 - "NUCLEAR GREEN" EDITION (Fixed Args)
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

def filter_pcb_aggressive(img, debug=False, out_prefix=""):
    H, W = img.shape[:2]
    
    # 1. Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Define Green Range
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([95, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    if debug: cv2.imwrite(f"{out_prefix}debug_1_raw_mask.png", mask)

    # 3. Aggressive Fusion
    k_size = int(max(W, H) * 0.02) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    if debug: cv2.imwrite(f"{out_prefix}debug_2_clean_mask.png", mask_clean)

    # 4. Find largest blob
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("[WARN] No green detected. Returning center crop.")
        return img[int(H*0.1):int(H*0.9), int(W*0.1):int(W*0.9)]

    c = max(contours, key=cv2.contourArea)
    
    # 5. Crop
    x, y, w, h = cv2.boundingRect(c)
    padding = 10 
    x_new = max(0, x - padding)
    y_new = max(0, y - padding)
    w_new = min(W - x_new, w + 2*padding)
    h_new = min(H - y_new, h + 2*padding)
    
    print(f"[CROP] Found PCB Area: {w}x{h} -> Crop: x={x_new}, y={y_new}, w={w_new}, h={h_new}")
    cropped = img[y_new:y_new+h_new, x_new:x_new+w_new]
    
    return cropped

def main():
    # SỬA LỖI Ở ĐÂY: Khai báo tham số rõ ràng
    parser = argparse.ArgumentParser(description="PCB Crop Recursive")
    parser.add_argument("input", type=str, help="Input image path")
    parser.add_argument("--output", "-o", type=str, required=False, help="Output image path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Dummy args để tương thích ngược (nếu có script cũ gọi)
    parser.add_argument("--min-area", type=int, default=0, help="Ignored")
    parser.add_argument("--padding", type=int, default=0, help="Ignored")
    
    args = parser.parse_args()
    
    img = cv2.imread(args.input)
    if img is None:
        print(f"[ERROR] Cannot read image: {args.input}")
        sys.exit(1)
    
    # Đường dẫn debug
    in_p = Path(args.input)
    debug_pre = str(in_p.parent / f"{in_p.stem}_") if args.debug else ""

    result = filter_pcb_aggressive(img, debug=args.debug, out_prefix=debug_pre)
    
    # Output path logic
    if args.output:
        out_path = args.output
    else:
        out_path = str(in_p.parent / f"{in_p.stem}_filtered.png")
        
    cv2.imwrite(out_path, result)
    print(f"[DONE] Saved: {out_path}")

if __name__ == "__main__":
    main()