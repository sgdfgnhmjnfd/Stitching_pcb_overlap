import cv2
import numpy as np
import argparse
from pathlib import Path

def auto_crop_smart(img_path, output_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[SKIP] {img_path}")
        return False

    h, w = img.shape[:2]
    original = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 60, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, 2)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge, 2)

    combined = cv2.bitwise_and(edges, edges, mask=mask_green)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_score = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < h * w * 0.08:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh)
        if aspect < 0.4 or aspect > 2.5:
            continue

        hull = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-6)
        if solidity < 0.7:
            continue

        score = area * solidity
        if score > best_score:
            best_score = score
            best_cnt = c

    if best_cnt is None:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            if cv2.contourArea(c) > h * w * 0.1:
                best_cnt = c
                break

    if best_cnt is None:
        print(f"[FAIL] {img_path}")
        return False

    x, y, bw, bh = cv2.boundingRect(best_cnt)
    pad = int(0.02 * max(bw, bh))
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)

    cropped = original[y:y+bh, x:x+bw]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cropped)
    print(f"[OK] {img_path}")
    return True


def main(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    imgs = list(input_root.rglob("*.png")) \
         + list(input_root.rglob("*.jpg")) \
         + list(input_root.rglob("*.jpeg")) \
         + list(input_root.rglob("*.bmp"))

    print(f"[INFO] Tổng ảnh: {len(imgs)}")

    for img_path in imgs:
        rel = img_path.relative_to(input_root)
        out_path = output_root / rel
        auto_crop_smart(img_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root")
    parser.add_argument("output_root")
    args = parser.parse_args()

    main(args.input_root, args.output_root)
