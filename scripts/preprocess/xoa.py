import shutil
import os
from pathlib import Path

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN CẦN QUÉT
# ==========================================
# Nó sẽ quét từ đây và chui vào tất cả thư mục con để tìm cái gì tên là "rotated"
ROOT_DIR = r"D:\Dataset_PCB_Final" 

def clean_rotated_cases(root_path):
    root = Path(root_path)
    if not root.exists():
        print(f"❌ Không tìm thấy đường dẫn: {root}")
        return

    print(f"🔍 Đang quét tìm 'rotated' trong: {root} ...")

    # 1. Tìm các THƯ MỤC tên là 'rotated' (Thư mục chứa tiles đầu vào)
    # rglob: tìm đệ quy tất cả folder con
    rotated_folders = [p for p in root.rglob("rotated") if p.is_dir()]

    # 2. Tìm các FILE kết quả có chữ 'rotated' trong tên (File ảnh kết quả)
    rotated_files = [p for p in root.rglob("*rotated*") if p.is_file()]

    total_items = len(rotated_folders) + len(rotated_files)

    if total_items == 0:
        print("✅ Sạch sẽ! Không tìm thấy file hay folder nào tên 'rotated'.")
        return

    print(f"\n⚠️ TÌM THẤY {total_items} MỤC LIÊN QUAN ĐẾN ROTATED:")
    
    if rotated_folders:
        print(f"\n--- [FOLDERS] ({len(rotated_folders)}) ---")
        for p in rotated_folders: print(f"  📁 {p}")
        print("  (Lưu ý: Xoá folder là xoá luôn tất cả ảnh con bên trong)")
        
    if rotated_files:
        print(f"\n--- [FILES] ({len(rotated_files)}) ---")
        for p in rotated_files: print(f"  📄 {p}")

    print("\n" + "!"*60)
    print("WARNING: Hành động này không thể hoàn tác!")
    confirm = input("🔥 Gõ 'yes' để xác nhận XOÁ VĨNH VIỄN tất cả các mục trên: ")
    print("!"*60 + "\n")
    
    if confirm.lower().strip() == "yes":
        # Xoá Files trước (nếu nó nằm ngoài folder rotated)
        for p in rotated_files:
            try:
                if p.exists():
                    os.remove(p)
                    print(f"🗑️ Đã xoá file: {p.name}")
            except Exception as e:
                print(f"❌ Lỗi xoá file {p.name}: {e}")

        # Xoá Folders
        for p in rotated_folders:
            try:
                if p.exists():
                    shutil.rmtree(p) # Xoá đệ quy
                    print(f"🗑️ Đã xoá folder: {p}")
            except Exception as e:
                print(f"❌ Lỗi xoá folder {p}: {e}")
        
        print("\n✅ ĐÃ DỌN DẸP SẠCH SẼ CASE ROTATED!")
    else:
        print("\n❌ Đã huỷ thao tác. Dữ liệu vẫn còn nguyên.")

if __name__ == "__main__":
    clean_rotated_cases(ROOT_DIR)