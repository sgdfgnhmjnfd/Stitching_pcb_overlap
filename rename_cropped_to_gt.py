"""
Rename all cropped.png to GT.png recursively
"""

from pathlib import Path
import os

TILES_ROOT = Path(r"D:\Dataset_PCB_Final\tiles")

def rename_files():
    count = 0
    
    # Find all cropped.png files recursively
    for cropped_file in TILES_ROOT.rglob("cropped.png"):
        new_name = cropped_file.parent / "GT.png"
        
        try:
            os.rename(str(cropped_file), str(new_name))
            print(f"✓ {cropped_file.relative_to(TILES_ROOT)} → GT.png")
            count += 1
        except Exception as e:
            print(f"✗ Error: {cropped_file} - {e}")
    
    print(f"\nDone! Renamed {count} files")

if __name__ == "__main__":
    rename_files()
