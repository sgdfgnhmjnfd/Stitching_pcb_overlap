import cv2
import numpy as np
from pathlib import Path
from collections import deque
import re
import sys
import time
from datetime import datetime
import argparse  # Thêm thư viện xử lý tham số dòng lệnh

# ==========================================
# 1. LOGGING & UTILS
# ==========================================
class StitchLogger:
    def __init__(self, verbose=True):
        self.verbose = verbose
    def log(self, level, message):
        if self.verbose: print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
    def progress(self, message, percentage):
        if self.verbose:
            sys.stdout.write(f"\r[{percentage}%] {message}")
            sys.stdout.flush()
            if percentage == 100: print()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path):
    if not folder.exists() or not folder.is_dir(): return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])

def parse_grid_tiles(folder: Path, logger=None):
    clean_path_str = str(folder).strip().strip('"').strip("'")
    folder_path = Path(clean_path_str)
    if not folder_path.exists(): return [], 0, 0, None

    tiles = []
    # Regex bắt rX_cY
    patterns = [re.compile(r'tile[_-]?r(\d+)[_-]?c(\d+)', re.IGNORECASE), re.compile(r'(\d+)[-_](\d+)')]
    files = list_images(folder_path)
    
    for f in files:
        r, c = -1, -1
        for pat in patterns:
            m = pat.search(f.stem)
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                break
        if r != -1 and c != -1:
            tiles.append({'r': r, 'c': c, 'path': f})
    
    if not tiles: return [], 0, 0, None
    max_r = max(t['r'] for t in tiles)
    max_c = max(t['c'] for t in tiles)
    tile_map = {(t['r'], t['c']): t['path'] for t in tiles}
    
    if logger: logger.log("INFO", f"Found {len(tiles)} tiles in {folder.name}")
    return tiles, max_r+1, max_c+1, tile_map

# ==========================================
# 2. ANALYSIS
# ==========================================
def analyze_geometry(tiles, tile_map, logger):
    img0 = cv2.imread(str(tiles[0]['path']))
    h, w = img0.shape[:2]
    sift = cv2.SIFT_create(nfeatures=2000)
    
    angles = []
    pairs = []
    for t in tiles:
        if len(pairs) > 50: break
        r, c = t['r'], t['c']
        if (r, c+1) in tile_map: pairs.append(((r,c), (r,c+1)))
        
    for idx1, idx2 in pairs:
        try:
            img1 = cv2.imread(str(tile_map[idx1]), 0)
            img2 = cv2.imread(str(tile_map[idx2]), 0)
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            if des1 is None or des2 is None: continue
            matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) > 10:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
                M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
                if M is not None:
                    ang = np.degrees(np.arctan2(M[1,0], M[0,0]))
                    if abs(ang) < 20: angles.append(ang)
        except: continue
        
    global_angle = np.median(angles) if angles else 0.0
    logger.log("INFO", f"Detected Global Angle: {global_angle:.2f} degrees")
    return global_angle, w, h

# ==========================================
# 3. THUẬT TOÁN 1: CHAIN STITCHING (Rotated)
# ==========================================
def solve_chain_stitching(rows, cols, tile_map, w, h, logger):
    logger.log("INFO", "Mode: CHAIN STITCHING")
    
    global_H = {(0,0): np.eye(3, dtype=np.float32)}
    queue = deque([(0,0)])
    visited = {(0,0)}
    
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    
    total = rows * cols
    count = 0
    
    while queue:
        u = queue.popleft()
        ur, uc = u
        
        neighbors = []
        if (ur, uc+1) in tile_map: neighbors.append(((ur, uc+1), 'H'))
        if (ur+1, uc) in tile_map: neighbors.append(((ur+1, uc), 'V'))
        
        for v, direction in neighbors:
            if v in visited: continue
            
            img1 = cv2.imread(str(tile_map[u]), 0)
            img2 = cv2.imread(str(tile_map[v]), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img1 = clahe.apply(img1)
            img2 = clahe.apply(img2)
            
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            
            H_rel = np.eye(3, dtype=np.float32)
            valid_match = False
            
            if des1 is not None and des2 is not None:
                matches = matcher.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                
                if len(good) > 10:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
                    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
                    
                    if M is not None:
                        scale = np.sqrt(M[0,0]**2 + M[0,1]**2)
                        if 0.9 < scale < 1.1:
                            M_3x3 = np.vstack([M, [0,0,1]])
                            H_rel = np.linalg.inv(M_3x3)
                            valid_match = True
            
            if not valid_match:
                if direction == 'H': H_rel[0, 2] = w * 0.9 
                else: H_rel[1, 2] = h * 0.9
            
            global_H[v] = global_H[u] @ H_rel
            visited.add(v)
            queue.append(v)
            count += 1
            if count % 5 == 0: logger.progress("Matching", int(count/total*90))
            
    return global_H

# ==========================================
# 4. THUẬT TOÁN 2: RIGID GRID (Noisy/Blur)
# ==========================================
def solve_grid_stitching(rows, cols, tile_map, w, h, logger):
    logger.log("INFO", "Mode: RIGID GRID")
    
    sift = cv2.SIFT_create(nfeatures=2000)
    steps_x, steps_y = [], []
    
    pairs = []
    for t in tile_map:
        r, c = t
        if (r, c+1) in tile_map: pairs.append(((r,c), (r,c+1), 'H'))
        if (r+1, c) in tile_map: pairs.append(((r,c), (r+1,c), 'V'))
        if len(pairs) > 50: break
        
    for idx1, idx2, direction in pairs:
        try:
            img1 = cv2.imread(str(tile_map[idx1]), 0)
            img2 = cv2.imread(str(tile_map[idx2]), 0)
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            if des1 is None or des2 is None: continue
            
            matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            
            if len(good) > 10:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
                diff = np.median(pts1 - pts2, axis=0) 
                if direction == 'H' and abs(diff[0]) > w*0.5: steps_x.append(abs(diff[0]))
                if direction == 'V' and abs(diff[1]) > h*0.5: steps_y.append(abs(diff[1]))
        except: continue
        
    step_x = np.median(steps_x) if steps_x else w * 0.85
    step_y = np.median(steps_y) if steps_y else h * 0.85
    
    global_H = {}
    for r in range(rows):
        for c in range(cols):
            if (r,c) not in tile_map: continue
            M = np.eye(3, dtype=np.float32)
            M[0, 2] = c * step_x
            M[1, 2] = r * step_y
            global_H[(r,c)] = M
            
    return global_H

# ==========================================
# 5. RENDER (FIX LỖ ĐEN)
# ==========================================
def render_safe(global_H, tile_map, w, h, out_file, logger):
    logger.progress("Rendering...", 90)
    
    all_corners = []
    base_corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
    
    for M in global_H.values():
        corners = cv2.perspectiveTransform(base_corners, M)
        all_corners.append(corners)
        
    all_pts = np.concatenate(all_corners, axis=0)
    xmin, ymin = np.int32(all_pts.min(axis=0).ravel())
    xmax, ymax = np.int32(all_pts.max(axis=0).ravel())
    
    W_canvas = int(xmax - xmin)
    H_canvas = int(ymax - ymin)
    
    if W_canvas * H_canvas > 2_000_000_000:
        logger.log("ERROR", "Canvas too huge!")
        return False
        
    canvas = np.zeros((H_canvas, W_canvas, 3), dtype=np.float32)
    weights = np.zeros((H_canvas, W_canvas), dtype=np.float32)
    
    T_off = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
    
    for (r,c), M in global_H.items():
        if (r,c) not in tile_map: continue
        img = cv2.imread(str(tile_map[(r,c)]))
        if img is None: continue
        
        Final_M = T_off @ M
        
        # Warp Ảnh
        warped_img = cv2.warpPerspective(img.astype(np.float32), Final_M, (W_canvas, H_canvas))
        
        # Warp Mask (HARD MASK - KHÔNG LÀM MỀM)
        mask = np.ones((h, w), dtype=np.float32)
        warped_mask = cv2.warpPerspective(mask, Final_M, (W_canvas, H_canvas))
        
        canvas += warped_img * warped_mask[..., None]
        weights += warped_mask
        
    # Chuẩn hóa
    mask = weights > 0.001
    canvas[mask] /= weights[mask, np.newaxis]
    final = np.clip(canvas, 0, 255).astype(np.uint8)
    
    try:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), final)
        logger.log("SUCCESS", f"Saved: {out_file.name}")
        return True
    except Exception as e:
        logger.log("ERROR", f"Save Error: {e}")
        return False

# ==========================================
# 6. MAIN CONTROLLER (ĐÃ SỬA: NHẬN ARGUMENTS)
# ==========================================
def process_single_run(tiles_dir, out_file, logger):
    tiles_dir = Path(tiles_dir)
    out_file = Path(out_file)
    
    logger.log("INFO", f"Input Tiles: {tiles_dir}")
    logger.log("INFO", f"Output Target: {out_file}")
    
    tiles, rows, cols, tile_map = parse_grid_tiles(tiles_dir, logger)
    if not tiles: 
        logger.log("ERROR", "No tiles found in directory!")
        return False
    
    angle, w, h = analyze_geometry(tiles, tile_map, logger)
    
    # HYBRID LOGIC
    if abs(angle) > 0.5:
        transforms = solve_chain_stitching(rows, cols, tile_map, w, h, logger)
    else:
        transforms = solve_grid_stitching(rows, cols, tile_map, w, h, logger)
        
    return render_safe(transforms, tile_map, w, h, out_file, logger)

if __name__ == "__main__":
    # KHỞI TẠO ARGPARSE ĐỂ NHẬN LỆNH TỪ APP C#
    parser = argparse.ArgumentParser(description="Stitch Tiles Single Run")
    parser.add_argument("--tiles_dir", type=str, required=True, help="Path to input tiles folder")
    parser.add_argument("--out_root", type=str, required=True, help="Full path for output image")
    
    # Các tham số phụ (để tương thích lệnh gọi cũ, có thể không dùng)
    parser.add_argument("--mode", type=str, default="stitch")
    parser.add_argument("--nfeatures", type=int, default=5000)
    parser.add_argument("--overlap", type=float, default=0.25)

    args = parser.parse_args()
    
    logger = StitchLogger()
    logger.log("START", "Initializing Stitching Engine...")
    
    try:
        success = process_single_run(args.tiles_dir, args.out_root, logger)
        if not success:
            sys.exit(1) # Trả về lỗi để App biết
    except Exception as e:
        logger.log("CRITICAL", f"Unhandled Exception: {e}")
        sys.exit(1)