#!/usr/bin/env python3
"""
EVALUATION V3: HIGH PRECISION
- Stage 1: SIFT Alignment (Rough)
- Stage 2: ECC Alignment (Fine-tune Sub-pixel)
- Stage 3: Histogram Matching (Fix Lighting)
"""

import cv2
import numpy as np
from pathlib import Path
import csv
import argparse
from skimage.metrics import structural_similarity as ssim

# ===============================
# PRE-PROCESSING
# ===============================

def match_histograms(source, reference):
    """
    Điều chỉnh màu sắc của source cho giống reference
    """
    # Chuyển sang LAB để chỉnh độ sáng (L channel)
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    
    src_l, src_a, src_b = cv2.split(src_lab)
    ref_l, ref_a, ref_b = cv2.split(ref_lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    src_l = clahe.apply(src_l)
    ref_l = clahe.apply(ref_l)

    # Đơn giản hóa: Match mean và stddev của kênh L
    src_mean, src_std = src_l.mean(), src_l.std()
    ref_mean, ref_std = ref_l.mean(), ref_l.std()
    
    src_l = ((src_l - src_mean) * (ref_std / (src_std + 1e-6)) + ref_mean)
    src_l = np.clip(src_l, 0, 255).astype(np.uint8)
    
    merged = cv2.merge((src_l, src_a, src_b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# ===============================
# ALIGNMENT MAGIC
# ===============================

def align_images_precision(im_source, im_target):
    # 1. SIFT (Thô)
    gray_src = cv2.cvtColor(im_source, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(im_target, cv2.COLOR_BGR2GRAY)
    
    h_tgt, w_tgt = gray_tgt.shape[:2]

    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(gray_src, None)
    kp2, des2 = sift.detectAndCompute(gray_tgt, None)

    if des1 is None or des2 is None: return cv2.resize(im_source, (w_tgt, h_tgt))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 10: return cv2.resize(im_source, (w_tgt, h_tgt))

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M_sift, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M_sift is None: return cv2.resize(im_source, (w_tgt, h_tgt))

    # Warp SIFT
    rough_aligned = cv2.warpPerspective(im_source, M_sift, (w_tgt, h_tgt))

    # 2. ECC (Tinh chỉnh - Enhanced Correlation Coefficient)
    # Bước này làm ảnh khớp chính xác từng pixel
    try:
        # Chuyển ảnh xám để align
        im1_gray = cv2.cvtColor(rough_aligned, cv2.COLOR_BGR2GRAY)
        im2_gray = gray_tgt

        # Ma trận biến đổi khởi tạo (Identity)
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        
        # Số lần lặp tối đa và ngưỡng dừng
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        
        # Chạy thuật toán (Warp Homography)
        # Lưu ý: ECC rất dễ fail nếu ảnh quá khác biệt, nên ta để trong try-catch
        (_, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria)
        
        # Áp dụng warp tinh chỉnh
        final_aligned = cv2.warpPerspective(rough_aligned, warp_matrix, (w_tgt, h_tgt), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        
        return final_aligned
    except Exception as e:
        # Nếu ECC fail, dùng kết quả SIFT
        # print(f"[WARN] ECC Alignment failed: {e}")
        return rough_aligned

# ===============================
# METRICS & MAIN
# ===============================

def calculate_metrics(img, gt):
    if img.shape != gt.shape:
        img = cv2.resize(img, (gt.shape[1], gt.shape[0]))

    # Histogram Matching (Chống lệch màu)
    img_fixed = match_histograms(img, gt)

    # MSE & PSNR
    mse = np.mean((img_fixed.astype(float) - gt.astype(float)) ** 2)
    psnr = 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

    # SSIM
    g1 = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    score_ssim = ssim(g1, g2)

    # Edge IoU (Thêm độ bao dung 1 chút)
    e1 = cv2.Canny(g1, 50, 150)
    e2 = cv2.Canny(g2, 50, 150)
    kernel = np.ones((3,3), np.uint8) # Dilate để chấp nhận lệch 1-2px
    e1 = cv2.dilate(e1, kernel)
    e2 = cv2.dilate(e2, kernel)
    
    inter = np.logical_and(e1, e2).sum()
    union = np.logical_or(e1, e2).sum()
    iou = inter / (union + 1e-6)

    return psnr, score_ssim, iou, img_fixed

def save_heatmap(img, gt, out_path):
    diff = cv2.absdiff(img, gt)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Tăng độ tương phản heatmap
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), heatmap)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_stitch", required=True)
    parser.add_argument("--gt_final", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--filtered", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--case", default="unknown")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = cv2.imread(args.raw)
    pcb = cv2.imread(args.filtered)
    gt_s = cv2.imread(args.gt_stitch)
    gt_f = cv2.imread(args.gt_final)

    if any(img is None for img in [raw, pcb, gt_s, gt_f]):
        return

    print(f"--- Evaluating Case: {args.case} ---")

    # 1. Evaluate Stitching
    print(f"1. Checking Stitch Quality...")
    raw_aligned = align_images_precision(raw, gt_s)
    psnr_s, ssim_s, iou_s, _ = calculate_metrics(raw_aligned, gt_s)

    # 2. Evaluate Final PCB
    print(f"2. Checking Final Quality...")
    pcb_aligned = align_images_precision(pcb, gt_f)
    psnr_f, ssim_f, iou_f, final_img_fixed = calculate_metrics(pcb_aligned, gt_f)

    # Save visual debug
    save_heatmap(final_img_fixed, gt_f, out / "heatmap_final.png")
    cv2.imwrite(str(out / "debug_aligned_final.png"), final_img_fixed)

    # CSV
    csv_path = out / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case", 
            "PSNR_STITCH", "SSIM_STITCH", "IOU_STITCH",
            "PSNR_FINAL", "SSIM_FINAL", "IOU_FINAL"
        ])
        writer.writerow([
            args.case,
            f"{psnr_s:.2f}", f"{ssim_s:.4f}", f"{iou_s:.4f}",
            f"{psnr_f:.2f}", f"{ssim_f:.4f}", f"{iou_f:.4f}"
        ])

    print(f"   > STITCH: PSNR={psnr_s:.2f} | SSIM={ssim_s:.3f}")
    print(f"   > FINAL : PSNR={psnr_f:.2f} | SSIM={ssim_f:.3f}")

if __name__ == "__main__":
    main()