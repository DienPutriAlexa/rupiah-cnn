"""
====================================================================
  LANGKAH 5: Grouping Nominal & Splitting Dataset (Train/Val/Test)
  Deteksi Mata Uang Rupiah - CNN Aksesibilitas
====================================================================

Fungsi:
  - Mengelompokkan 30 subfolder (D/B & tahun) menjadi 8 kelas nominal
  - Membagi dataset secara acak dan seimbang:
      * Train : 70%
      * Val   : 15%
      * Test  : 15%
  - Menyalin file ke folder tujuan tanpa mengubah folder asli.

Cara pakai:
  python split_dataset.py --source_dir "D:/path/ke/rupiah-banknotes" --output_dir "D:/path/ke/rupiah-cnn/dataset_split"
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

# Mapping kode folder ke nama kelas nominal yang ramah
NOMINAL_MAPPING = {
    "001": "1000",
    "002": "2000",
    "005": "5000",
    "010": "10000",
    "020": "20000",
    "050": "50000",
    "075": "75000",
    "100": "100000"
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_nominal(folder_name: str) -> str:
    """Mengekstrak nominal dari format folder contoh '2016-010D' -> '10000'"""
    parts = folder_name.split("-")
    if len(parts) >= 2:
        code = parts[1][:3] # ambil 3 digit kode (001, 010, dsb)
        if code in NOMINAL_MAPPING:
            return NOMINAL_MAPPING[code]
    return "UNKNOWN"

def split_and_copy(source_dir: Path, output_dir: Path, train_ratio=0.70, val_ratio=0.15, seed=42):
    random.seed(seed)
    
    print("\n" + "="*60)
    print("  📦 MEMPROSES & MEMBAGI DATASET RUPIAH")
    print("="*60)
    print(f"Folder Sumber : {source_dir}")
    print(f"Folder Tujuan : {output_dir}")
    print(f"Rasio Split   : Train {int(train_ratio*100)}% | Val {int(val_ratio*100)}% | Test {int((1-train_ratio-val_ratio)*100)}%\n")

    # Kumpulkan semua file per nominal
    nominal_files = {nominal: [] for nominal in NOMINAL_MAPPING.values()}
    
    for subfolder in source_dir.iterdir():
        if subfolder.is_dir():
            nominal = parse_nominal(subfolder.name)
            if nominal in nominal_files:
                images = [f for f in subfolder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
                nominal_files[nominal].extend(images)

    # Buat direktori train, val, test
    splits = ["train", "val", "test"]
    for s in splits:
        for nominal in NOMINAL_MAPPING.values():
            (output_dir / s / nominal).mkdir(parents=True, exist_ok=True)

    summary = {s: {nominal: 0 for nominal in NOMINAL_MAPPING.values()} for s in splits}

    for nominal, files in nominal_files.items():
        random.shuffle(files)
        total = len(files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # Salin file ke masing-masing split
        for f in tqdm(train_files, desc=f"  [Train] {nominal:<6}", unit="img", ncols=75):
            shutil.copy2(f, output_dir / "train" / nominal / f"{f.parent.name}_{f.name}")
            summary["train"][nominal] += 1

        for f in tqdm(val_files, desc=f"  [Val]   {nominal:<6}", unit="img", ncols=75):
            shutil.copy2(f, output_dir / "val" / nominal / f"{f.parent.name}_{f.name}")
            summary["val"][nominal] += 1

        for f in tqdm(test_files, desc=f"  [Test]  {nominal:<6}", unit="img", ncols=75):
            shutil.copy2(f, output_dir / "test" / nominal / f"{f.parent.name}_{f.name}")
            summary["test"][nominal] += 1

    # Tampilkan Ringkasan
    print("\n" + "="*60)
    print("  📊 RINGKASAN HASIL SPLIT DATASET")
    print("="*60)
    print(f"{'Nominal':<12} {'Train (70%)':>12} {'Val (15%)':>12} {'Test (15%)':>12} {'Total':>10}")
    print("-" * 60)
    total_all = 0
    for nominal in sorted(NOMINAL_MAPPING.values(), key=lambda x: int(x)):
        tr = summary["train"][nominal]
        vl = summary["val"][nominal]
        ts = summary["test"][nominal]
        subtotal = tr + vl + ts
        total_all += subtotal
        print(f"Rp {nominal:<9} {tr:>12} {vl:>12} {ts:>12} {subtotal:>10}")
    
    print("-" * 60)
    tr_tot = sum(summary["train"].values())
    vl_tot = sum(summary["val"].values())
    ts_tot = sum(summary["test"].values())
    print(f"{'TOTAL':<12} {tr_tot:>12} {vl_tot:>12} {ts_tot:>12} {total_all:>10}")
    print("\n✅ Dataset berhasil disiapkan di folder tujuan!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Dataset Rupiah ke Train/Val/Test")
    parser.add_argument("--source_dir", type=str, default=".", help="Folder dataset asli (rupiah-banknotes)")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder tujuan dataset_split")
    args = parser.parse_args()

    split_and_copy(Path(args.source_dir), Path(args.output_dir))
