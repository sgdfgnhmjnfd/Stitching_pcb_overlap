import subprocess
from pathlib import Path
import time
import concurrent.futures
import threading
import multiprocessing

# ================= CONFIG =================
ROOT = Path("D:/Dataset_PCB_Final")
SCRIPT = ROOT / "scripts/stitching/stitch_tiles.py"
OUTPUT = ROOT / "results/stitched"

CASES = ["base", "shuffled", "noisy", "blurred", "dazzled", 
         "rotated", "rotated_small", "missing", "combined"]

# Số lượng job chạy cùng lúc. 
# Vì stitch_tiles.py đã dùng đa nhân, nên set cái này thấp thôi (2-3) để tránh treo máy.
MAX_CONCURRENT_JOBS = 3 
# ==========================================

# Khóa để in ra màn hình không bị lộn xộn khi chạy song song
print_lock = threading.Lock()

def get_all_machines(root):
    tiles_root = root / "tiles"
    machines = []
    if not tiles_root.exists():
        print(f"[ERROR] Không tìm thấy thư mục: {tiles_root}")
        return []
        
    for machine_dir in tiles_root.iterdir():
        if not machine_dir.is_dir():
            continue
        for type_dir in machine_dir.iterdir():
            if not type_dir.is_dir():
                continue
            for case_dir in type_dir.iterdir():
                if case_dir.is_dir() and (case_dir / "base").exists():
                    rel_path = case_dir.relative_to(tiles_root)
                    machines.append(str(rel_path))
    return sorted(set(machines))

def get_params(case):
    # --- TỐI ƯU HÓA THAM SỐ ---
    # Giảm số lượng features xuống mức hợp lý (4000-6000) để khớp với thuật toán mới
    # Tăng nhẹ ratio hoặc giữ nguyên để đảm bảo độ chính xác
    
    if case == "combined":
        return {"orb_features": "6000", "min_matches": "8", "ratio": "0.75"}
    elif case in ["rotated", "rotated_small"]:
        # Case xoay cần nhiều điểm hơn chút, nhưng 12000 là quá thừa
        return {"orb_features": "6000", "min_matches": "15", "ratio": "0.7"}
    elif case in ["dazzled", "blurred"]:
        return {"orb_features": "5000", "min_matches": "12", "ratio": "0.75"}
    else:
        # Case dễ (base, shuffled...) chỉ cần 4000 là đủ nhanh và chuẩn
        return {"orb_features": "4000", "min_matches": "10", "ratio": "0.75"}

def process_case_wrapper(args):
    """Wrapper function để unpack arguments cho Executor"""
    return process_case(*args)

def process_case(machine, case, task_id, total):
    name = f"{machine}/{case}"
    tiles_dir = ROOT / "tiles" / machine / case
    
    # Format ngắn gọn hơn
    short_name = f".../{machine.split('/')[-1]}/{case}"
    
    if not tiles_dir.exists():
        with print_lock:
            print(f"[{task_id}/{total}] [SKIP] {name}")
        return name, "skipped", None
    
    params = get_params(case)
    
    cmd = [
        "python", str(SCRIPT),
        "--tiles_dir", str(tiles_dir),
        "--out_root", str(OUTPUT),
        "--blend",
        "--orb_features", params["orb_features"],
        "--min_matches", params["min_matches"],
        "--ratio", params["ratio"]
    ]
    
    start_time = time.time()
    
    # In ra log bắt đầu (dùng lock để không bị đè dòng)
    with print_lock:
        print(f"[{task_id}/{total}] ▶ START: {short_name}")

    try:
        # Chạy subprocess
        # capture_output=True để ẩn đống log dài dòng của script con, chỉ hiện kết quả cuối
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start_time
        
        with print_lock:
            if result.returncode == 0:
                print(f"[{task_id}/{total}] [✓] DONE ({elapsed:.1f}s): {short_name}")
                return name, "success", None
            else:
                print(f"[{task_id}/{total}] [✗] FAIL ({elapsed:.1f}s): {short_name}")
                # In ra vài dòng lỗi cuối cùng để debug
                error_log = "\n".join(result.stdout.splitlines()[-5:])
                print(f"    Error details: {error_log}")
                return name, "failed", f"Code {result.returncode}"
    
    except subprocess.TimeoutExpired:
        with print_lock:
            print(f"[{task_id}/{total}] [⏱] TIMEOUT: {short_name}")
        return name, "timeout", "Exceeded 10min"
    except Exception as e:
        with print_lock:
            print(f"[{task_id}/{total}] [!] ERROR: {short_name} - {e}")
        return name, "error", str(e)

# ================= MAIN =================
if __name__ == "__main__":
    ALL_MACHINES = get_all_machines(ROOT)

    print("="*70)
    print(f"BATCH PROCESSING (PARALLEL OPTIMIZED)")
    print("="*70)
    print(f"Input Root:  {ROOT}")
    print(f"Output Root: {OUTPUT}")
    print(f"Parallel Jobs: {MAX_CONCURRENT_JOBS}")
    print(f"Total Cases: {len(ALL_MACHINES)} machines x {len(CASES)} cases")
    print("="*70)
    
    # Tạo danh sách tasks
    tasks = []
    task_counter = 0
    total_tasks = len(ALL_MACHINES) * len(CASES)
    
    for machine in ALL_MACHINES:
        for case in CASES:
            task_counter += 1
            tasks.append((machine, case, task_counter, total_tasks))
            
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_list = []

    start_batch = time.time()

    # CHẠY SONG SONG
    # Sử dụng ThreadPoolExecutor vì process_case chủ yếu là chờ subprocess (IO bound)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        # Submit tất cả tasks
        futures = {executor.submit(process_case_wrapper, t): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            name, status, error = future.result()
            
            if status == "success":
                success_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                failed_count += 1
                failed_list.append((name, error))

    total_time = time.time() - start_batch

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"✓ Success:  {success_count}/{total_tasks} ({success_count/total_tasks*100:.1f}%)")
    print(f"✗ Failed:   {failed_count}/{total_tasks} ({failed_count/total_tasks*100:.1f}%)")
    print(f"○ Skipped:  {skipped_count}/{total_tasks} ({skipped_count/total_tasks*100:.1f}%)")

    if failed_list:
        print("\n--- Failed Cases ---")
        for name, error in failed_list:
            print(f" • {name}")
            if error:
                print(f"    → {error}")

    print("\n[DONE]")