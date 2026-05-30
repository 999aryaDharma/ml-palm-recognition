"""
Script untuk mengorganisir dataset palm biometrics berdasarkan palm ID.

Naming Convention:
- 00001~00010: palm_00001
- 00011~00020: palm_00002
- 00021~00030: palm_00003
- ... dst sampai palm_00600

Setiap session (session1, session2) akan diorganisir dalam struktur folder:
session1/
├── palm_00001/
├── palm_00002/
└── ...
"""

import os
import shutil
from pathlib import Path

def get_palm_id(image_number):
    """
    Konversi nomor image ke palm ID.
    
    Image 00001-00010 -> palm_00001
    Image 00011-00020 -> palm_00002
    ... dst
    """
    # Kurangi 1 untuk membuat berbasis 0, bagi 10, tambah 1 untuk kembali berbasis 1
    palm_id = (image_number - 1) // 10 + 1
    return f"palm_{palm_id:05d}"

def organize_session(session_path, dry_run=False):
    """
    Mengorganisir images dalam session folder.
    
    Args:
        session_path: Path ke folder session (e.g., 'session1')
        dry_run: Jika True, hanya print tanpa membuat folder
    """
    session_path = Path(session_path)
    
    if not session_path.exists():
        print(f"ERROR: Path {session_path} tidak ditemukan!")
        return False
    
    # Cari semua .tiff files
    tiff_files = sorted(session_path.glob("*.tiff"))
    
    if not tiff_files:
        print(f"WARNING: Tidak ada .tiff files di {session_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Mengorganisir: {session_path.name}")
    print(f"Total images: {len(tiff_files)}")
    print(f"{'='*60}")
    
    # Group images by palm_id
    palm_groups = {}
    
    for tiff_file in tiff_files:
        # Extract image number dari filename (e.g., "00001.tiff" -> 1)
        image_number = int(tiff_file.stem)
        palm_id = get_palm_id(image_number)
        
        if palm_id not in palm_groups:
            palm_groups[palm_id] = []
        
        palm_groups[palm_id].append(tiff_file)
    
    print(f"Total palm IDs: {len(palm_groups)}")
    
    # Create folders dan move files
    for palm_id in sorted(palm_groups.keys()):
        palm_dir = session_path / palm_id
        
        if not dry_run:
            # Create folder jika belum ada
            palm_dir.mkdir(exist_ok=True)
            
            # Move files ke palm folder
            for image_file in palm_groups[palm_id]:
                dest_file = palm_dir / image_file.name
                
                if dest_file.exists():
                    print(f"  SKIP: {image_file.name} (sudah ada di {palm_id})")
                else:
                    shutil.move(str(image_file), str(dest_file))
        
        print(f"  {palm_id}: {len(palm_groups[palm_id])} images")
    
    print(f"\nOrganisasi selesai!")
    return True

def main():
    """Main function"""
    import sys
    
    # Paths
    base_path = Path(__file__).parent.parent / "data" / "raw"
    sessions = [base_path / "session1", base_path / "session2"]
    
    print("\n" + "="*60)
    print("PALM BIOMETRICS DATASET ORGANIZER")
    print("="*60)
    print(f"Base path: {base_path}")
    print(f"Sessions to organize: session1, session2")
    
    # Confirm before proceeding
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        print("\n[DRY RUN MODE] - Tidak ada file yang akan dipindahkan")
    else:
        confirm = input("\nLanjutkan mengorganisir dataset? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Dibatalkan!")
            return
    
    # Process each session
    for session_path in sessions:
        if session_path.exists():
            organize_session(session_path, dry_run=dry_run)
        else:
            print(f"Session {session_path.name} tidak ditemukan!")
    
    print("\n" + "="*60)
    print("SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()
