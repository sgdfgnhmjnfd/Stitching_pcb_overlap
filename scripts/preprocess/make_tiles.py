import cv2
import numpy as np
from pathlib import Path

# ========= CONFIG =========
RAW_ROOT = Path(r"D:\Dataset_PCB_Final\raw\mach1_good")
TILES_ROOT = Path(r"D:\Dataset_PCB_Final\tiles")

ROWS = 8
COLS = 8
OVERLAP = 0.25
# ==========================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def detect_and_crop_pcb(img):
    """Return full image without cropping"""
    return img

def split_with_overlap(img, rows, cols, overlap):
    """Split image into tiles with overlap, ensuring all tiles have same size"""
    h, w = img.shape[:2]
    
    # Calculate tile size to fit exactly
    # Formula: (rows-1)*step + tile_size = image_size
    # Where: step = tile_size * (1 - overlap)
    # Solving: tile_size = image_size / (1 + (rows-1)*(1-overlap))
    
    tile_h = int(h / (1 + (rows - 1) * (1 - overlap)))
    tile_w = int(w / (1 + (cols - 1) * (1 - overlap)))
    
    step_h = int(tile_h * (1 - overlap))
    step_w = int(tile_w * (1 - overlap))
    
    print(f"  Tile size: {tile_w}x{tile_h}, Step: {step_w}x{step_h}")
    
    tiles = []
    
    for r in range(rows):
        for c in range(cols):
            # For last row/col, adjust to fit exactly
            if r == rows - 1:
                y = h - tile_h
            else:
                y = r * step_h
            
            if c == cols - 1:
                x = w - tile_w
            else:
                x = c * step_w
            
            tile = img[y:y+tile_h, x:x+tile_w]
            
            # Verify size
            if tile.shape[0] != tile_h or tile.shape[1] != tile_w:
                print(f"[WARN] Tile ({r},{c}) wrong size: {tile.shape[:2]}, expected ({tile_h},{tile_w})")
                # Pad if needed
                if tile.shape[0] < tile_h or tile.shape[1] < tile_w:
                    padded = np.zeros((tile_h, tile_w, 3), dtype=img.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
            
            tiles.append((r, c, tile))
    
    return tiles

def process_image(img_path: Path, out_base: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Cannot read: {img_path}")
        return
    
    print(f"Processing {img_path.name}:")
    print(f"  Original size: {img.shape[1]}x{img.shape[0]}")
    
    # Step 1: Detect and crop PCB
    cropped = detect_and_crop_pcb(img)
    
    # Step 2: Split into tiles
    tiles = split_with_overlap(cropped, ROWS, COLS, OVERLAP)
    
    print(f"  Generated {len(tiles)}/{ROWS*COLS} tiles")
    
    # Save tiles
    for r, c, tile in tiles:
        name = f"tile_r{r}_c{c}.png"
        cv2.imwrite(str(out_base / name), tile)
    
    # Also save cropped PCB for reference
    cropped_path = out_base.parent / "cropped.png"
    cv2.imwrite(str(cropped_path), cropped)

def main():
    for mach_dir in RAW_ROOT.iterdir():
        if not mach_dir.is_dir():
            continue

        for img_path in mach_dir.glob("*.png"):
            out_base = (
                TILES_ROOT /
                mach_dir.name /
                img_path.stem /
                "base"
            )

            ensure_dir(out_base)
            process_image(img_path, out_base)

            print(f"[✓] {mach_dir.name}/{img_path.name} → base tiles\n")

if __name__ == "__main__":
    main()