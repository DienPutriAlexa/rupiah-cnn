"""
=======================================================
  LANGKAH 3 & 4: Inspeksi & Audit Dataset Rupiah
  Deteksi Mata Uang Rupiah - CNN Aksesibilitas
=======================================================

Fungsi:
  - Memverifikasi struktur folder dataset
  - Mendeteksi & menghapus gambar rusak (corrupt)
  - Menghitung distribusi jumlah gambar per kelas
  - Menyimpan laporan audit ke file CSV & TXT

Cara pakai:
  python inspect_dataset.py --dataset_dir "D:/path/ke/dataset"
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ─── Konfigurasi ──────────────────────────────────────────────────
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_WIDTH        = 32    # pixel minimum yang dianggap valid
MIN_HEIGHT       = 32
OUTPUT_REPORT    = "dataset_audit_report.txt"
OUTPUT_CSV       = "distribusi_kelas.csv"
# ──────────────────────────────────────────────────────────────────


def verifikasi_struktur(dataset_dir: Path):
    """Tampilkan struktur folder dataset."""
    print("\n" + "="*55)
    print("VERIFIKASI STRUKTUR FOLDER DATASET")
    print("="*55)

    if not dataset_dir.exists():
        print(f"[ERROR] Folder '{dataset_dir}' tidak ditemukan!")
        print("        Pastikan dataset sudah diekstrak dengan benar.")
        sys.exit(1)

    kelas = [d for d in sorted(dataset_dir.iterdir()) if d.is_dir()]
    if not kelas:
        print(f"[WARNING] Tidak ada subfolder kelas di dalam '{dataset_dir}'.")
        print("          Mungkin dataset perlu di-unzip lebih dalam.")
        return []

    print(f"\nFolder dataset : {dataset_dir}")
    print(f"Total kelas    : {len(kelas)}\n")
    for k in kelas:
        total = sum(1 for f in k.iterdir() if f.suffix.lower() in VALID_EXTENSIONS)
        print(f"{k.name:<20} → {total} gambar")
    return kelas


def inspeksi_gambar(kelas_dirs: list, hapus_rusak: bool = False):
    """
    Iterasi seluruh gambar, deteksi yang rusak atau tidak relevan.
    Jika hapus_rusak=True, file rusak akan dihapus otomatis.
    """
    print("\n" + "="*55)
    print("INSPEKSI & FILTER GAMBAR RUSAK")
    print("="*55)

    total_gambar   = 0
    gambar_rusak   = 0
    gambar_valid   = 0
    daftar_rusak   = []

    for kelas_dir in kelas_dirs:
        files = [f for f in kelas_dir.iterdir() if f.is_file()]
        for f in tqdm(files, desc=f"  {kelas_dir.name}", unit="img", ncols=70):
            # Lewati file bukan gambar
            if f.suffix.lower() not in VALID_EXTENSIONS:
                continue

            total_gambar += 1
            rusak = False
            alasan = ""

            try:
                with Image.open(f) as img:
                    img.verify()   # cek integritas file

                # Buka ulang untuk cek dimensi (verify() menutup file)
                with Image.open(f) as img:
                    w, h = img.size
                    if w < MIN_WIDTH or h < MIN_HEIGHT:
                        rusak  = True
                        alasan = f"Dimensi terlalu kecil ({w}x{h})"

            except Exception as e:
                rusak  = True
                alasan = f"File rusak/corrupt: {e}"

            if rusak:
                gambar_rusak += 1
                daftar_rusak.append({"path": str(f), "alasan": alasan})
                if hapus_rusak:
                    f.unlink()
                    print(f"\n    Dihapus: {f.name} | {alasan}")
                else:
                    print(f"\n    {f.name} | {alasan}")
            else:
                gambar_valid += 1

    print(f"\n  Total gambar diperiksa : {total_gambar}")
    print(f"  Gambar valid        : {gambar_valid}")
    print(f"  Gambar rusak        : {gambar_rusak}")
    if gambar_rusak > 0 and not hapus_rusak:
        print(" Jalankan ulang dengan --hapus untuk menghapus file rusak secara otomatis.")

    return daftar_rusak, total_gambar, gambar_valid, gambar_rusak


def hitung_distribusi(kelas_dirs: list):
    """Hitung jumlah gambar per kelas & tampilkan distribusi."""
    print("\n" + "="*55)
    print("   DISTRIBUSI DATA PER KELAS")
    print("="*55)

    distribusi = {}
    for kelas_dir in kelas_dirs:
        jumlah = sum(
            1 for f in kelas_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        )
        distribusi[kelas_dir.name] = jumlah

    total = sum(distribusi.values())
    rata  = total / len(distribusi) if distribusi else 0
    min_k = min(distribusi, key=distribusi.get) if distribusi else "-"
    max_k = max(distribusi, key=distribusi.get) if distribusi else "-"

    print(f"\n{'Kelas':<25} {'Jumlah':>8} {'% dari Total':>14}")
    print("-" * 50)
    for kelas, jml in sorted(distribusi.items()):
        persen = (jml / total * 100) if total > 0 else 0
        flag   = "  (<200)" if jml < 200 else ""
        print(f"  {kelas:<23} {jml:>8} {persen:>13.1f}%{flag}")

    print("-" * 50)
    print(f"  {'TOTAL':<23} {total:>8} {'100.0%':>14}")
    print(f"\n  Rata‑rata per kelas : {rata:.0f} gambar")
    print(f"  Kelas terbanyak     : {max_k} ({distribusi.get(max_k,0)} gambar)")
    print(f"  Kelas tersedikit    : {min_k} ({distribusi.get(min_k,0)} gambar)")

    # Cek keseimbangan: variance ≤ 10% dari rata-rata
    if rata > 0:
        variance_ok = all(abs(v - rata) / rata <= 0.10 for v in distribusi.values())
        status = " SEIMBANG (variance ≤ 10%)" if variance_ok else "  TIDAK SEIMBANG – pertimbangkan augmentasi atau oversampling"
        print(f"  Status distribusi   : {status}")

    return distribusi, total


def simpan_laporan(dataset_dir: Path, distribusi: dict, total: int, daftar_rusak: list,
                   gambar_valid: int, gambar_rusak: int):
    """Simpan laporan audit ke .txt dan distribusi ke .csv."""
    # Simpan laporan teks
    report_path = dataset_dir / OUTPUT_REPORT
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*55 + "\n")
        f.write("  LAPORAN AUDIT DATASET RUPIAH-CNN\n")
        f.write("="*55 + "\n\n")
        f.write(f"Folder dataset : {dataset_dir}\n")
        f.write(f"Total kelas    : {len(distribusi)}\n")
        f.write(f"Total gambar   : {total}\n")
        f.write(f"Gambar valid   : {gambar_valid}\n")
        f.write(f"Gambar rusak   : {gambar_rusak}\n\n")
        f.write("Distribusi per kelas:\n")
        for k, v in sorted(distribusi.items()):
            f.write(f"  {k}: {v} gambar\n")
        if daftar_rusak:
            f.write("\nDaftar Gambar Rusak:\n")
            for item in daftar_rusak:
                f.write(f"  {item['path']} | {item['alasan']}\n")

    # Simpan CSV distribusi
    csv_path = dataset_dir / OUTPUT_CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["kelas", "jumlah", "persen"])
        writer.writeheader()
        for k, v in sorted(distribusi.items()):
            persen = round(v / total * 100, 2) if total > 0 else 0
            writer.writerow({"kelas": k, "jumlah": v, "persen": persen})

    print(f"\n   Laporan disimpan  : {report_path}")
    print(f"   CSV distribusi    : {csv_path}")


# ─── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspeksi & Audit Dataset Rupiah")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="Path ke folder dataset yang berisi subfolder kelas"
    )
    parser.add_argument(
        "--hapus",
        action="store_true",
        help="Hapus gambar rusak secara otomatis"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Langkah 3a: Verifikasi struktur folder
    kelas_dirs = verifikasi_struktur(dataset_dir)

    if kelas_dirs:
        # Langkah 3b: Inspeksi & filter gambar rusak
        daftar_rusak, total_diperiksa, gambar_valid, gambar_rusak = inspeksi_gambar(
            kelas_dirs, hapus_rusak=args.hapus
        )

        # Langkah 4: Hitung distribusi kelas
        distribusi, total = hitung_distribusi(kelas_dirs)

        # Simpan laporan
        simpan_laporan(dataset_dir, distribusi, total, daftar_rusak,
                       gambar_valid, gambar_rusak)

    print("\n" + "="*55)
    print("   SELESAI! Cek laporan audit di folder dataset.")
    print("="*55 + "\n")
