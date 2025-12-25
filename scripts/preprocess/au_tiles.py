import cv2
import numpy as np
import random
import shutil
import argparse
from pathlib import Path

# ========= DEFAULT CONFIG =========
NOISE_STD = 20
MISSING_RATIO = 0.15
BLUR_K = 11
GLARE_ALPHA = 0.6
ROTATION_PROB = 0.3  # 30% tiles will be rotated
# ==================================

def clean_dir(p):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)

def load_tiles(base_dir):
    return sorted([p for p in base_dir.iterdir() if p.suffix.lower() == ".png"])

# ---------- Effects ----------
def add_noise(img, std=NOISE_STD):
    n = np.random.normal(0, std, img.shape)
    return np.clip(img + n, 0, 255).astype(np.uint8)

def add_blur(img, kernel_size=BLUR_K):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def add_glare(img, alpha=GLARE_ALPHA):
    h, w = img.shape[:2]
    overlay = np.zeros_like(img)
    cx, cy = random.randint(0, w), random.randint(0, h)
    r = random.randint(min(w, h) // 4, min(w, h) // 2)
    cv2.circle(overlay, (cx, cy), r, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

def rotate_90(img):
    return random.choice([
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img, cv2.ROTATE_180),
        cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ])

def rotate_small(img, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# ---------- Main Processing ----------
def process_case(case_dir, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    base = case_dir / "base"
    tiles = load_tiles(base)

    if len(tiles) == 0:
        print(f"[SKIP] No tiles in {base}")
        return

    print(f"\n[Processing] {case_dir} ({len(tiles)} tiles)")


    # --- 2. Noisy ---
    out = case_dir / "noisy"
    clean_dir(out)
    for p in tiles:
        img = cv2.imread(str(p))
        cv2.imwrite(str(out / p.name), add_noise(img))
    print(f"  [✓] noisy: {len(tiles)} tiles")

    # --- 3. Blurred ---
    out = case_dir / "blurred"
    clean_dir(out)
    for p in tiles:
        img = cv2.imread(str(p))
        cv2.imwrite(str(out / p.name), add_blur(img))
    print(f"  [✓] blurred: {len(tiles)} tiles")

    # --- 4. Glare ---
    out = case_dir / "dazzled"
    clean_dir(out)
    for p in tiles:
        img = cv2.imread(str(p))
        cv2.imwrite(str(out / p.name), add_glare(img))
    print(f"  [✓] dazzled: {len(tiles)} tiles")



    # --- 6. Small rotation ---
    out = case_dir / "rotated_small"
    clean_dir(out)
    for p in tiles:
        img = cv2.imread(str(p))
        img = rotate_small(img, max_angle=10)
        cv2.imwrite(str(out / p.name), img)
    print(f"  [✓] rotated_small")

    # --- 7. Missing tiles ---
    out = case_dir / "missing"
    clean_dir(out)
    keep_count = int(len(tiles) * (1 - MISSING_RATIO))
    kept_tiles = random.sample(tiles, keep_count)
    for p in kept_tiles:
        shutil.copy(p, out / p.name)
    print(f"  [✓] missing: {keep_count}/{len(tiles)} tiles kept")

    # --- 8. Combined corruption ---
    out = case_dir / "combined"
    clean_dir(out)
    for p in tiles:
        img = cv2.imread(str(p))
        if random.random() < 0.5:
            img = add_noise(img, std=15)
        if random.random() < 0.3:
            img = add_blur(img, kernel_size=7)
        if random.random() < 0.2:
            img = add_glare(img, alpha=0.4)
        cv2.imwrite(str(out / p.name), img)
    print(f"  [✓] combined")

def main():
    parser = argparse.ArgumentParser(description="Generate augmented tile datasets")
    parser.add_argument("--root", type=str,
                       default=r"D:\Dataset_PCB_Final\tiles",
                       help="Root directory containing tile folders")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", nargs="+",
                       help="Specific cases: mach1/mach1_good/2-1")
    args = parser.parse_args()

    root = Path(args.root)

    if args.cases:
        for case_path in args.cases:
            case_dir = root / case_path
            if case_dir.exists() and (case_dir / "base").exists():
                process_case(case_dir, seed=args.seed)
            else:
                print(f"[SKIP] {case_path}: not found or missing base")
    else:
        processed = 0

        for machine_dir in root.iterdir():
            if not machine_dir.is_dir():
                continue

            for group_dir in machine_dir.iterdir():
                if not group_dir.is_dir():
                    continue

                for case_dir in group_dir.iterdir():
                    if not case_dir.is_dir():
                        continue

                    if (case_dir / "base").exists():
                        process_case(case_dir, seed=args.seed)
                        processed += 1

        print(f"\n[DONE] Processed {processed} cases")


if __name__ == "__main__":
    main()
