"""
====================================================================
  LANGKAH 5.5: Generate Dataset "bukan_uang" (Background / Non-Money)
  Menambah kelas ke-9 agar CNN bisa membedakan uang vs bukan uang
====================================================================

Fungsi:
  - Membuat gambar sintetis yang bukan uang (gradien warna, noise,
    pola acak, warna solid) untuk melatih CNN mengenali "bukan uang"
  - Menyimpan ke dataset_split/train/bukan_uang/
                  dataset_split/val/bukan_uang/
                  dataset_split/test/bukan_uang/

Cara Pakai:
  python generate_background.py
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Jumlah gambar yang di-generate (disesuaikan agar seimbang dengan
# kelas uang terkecil yaitu 75000 yang punya 840 train / 180 val / 180 test)
NUM_TRAIN = 840
NUM_VAL   = 180
NUM_TEST  = 180
IMG_SIZE  = 128


def random_solid(size):
    """Gambar warna solid acak (simulasi tembok, meja, lantai)"""
    color = tuple(np.random.randint(0, 256, 3))
    img = Image.new("RGB", (size, size), color)
    return img


def random_gradient(size):
    """Gambar gradien dua warna (simulasi permukaan meja/kain)"""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    c1 = np.random.randint(0, 256, 3)
    c2 = np.random.randint(0, 256, 3)
    vertical = np.random.choice([True, False])
    for i in range(size):
        ratio = i / size
        color = (c1 * (1 - ratio) + c2 * ratio).astype(np.uint8)
        if vertical:
            arr[i, :] = color
        else:
            arr[:, i] = color
    return Image.fromarray(arr)


def random_noise(size):
    """Gambar noise acak (simulasi blur kamera tanpa objek jelas)"""
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    # Blur supaya tidak terlalu tajam (lebih realistis)
    img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(1, 4)))
    return img


def random_shapes(size):
    """Gambar dengan bentuk geometris acak (simulasi objek non-uang)"""
    bg_color = tuple(np.random.randint(0, 256, 3))
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    num_shapes = np.random.randint(2, 8)
    for _ in range(num_shapes):
        shape_type = np.random.choice(["rect", "ellipse", "line"])
        color = tuple(np.random.randint(0, 256, 3))
        x1, y1 = np.random.randint(0, size, 2)
        x2, y2 = np.random.randint(0, size, 2)
        coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if shape_type == "rect":
            draw.rectangle(coords, fill=color)
        elif shape_type == "ellipse":
            draw.ellipse(coords, fill=color)
        else:
            draw.line(coords, fill=color, width=np.random.randint(1, 6))
    return img


def random_texture(size):
    """Gambar pola berulang/tekstur (simulasi kain, lantai keramik)"""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    base_color = np.random.randint(50, 200, 3)
    arr[:] = base_color
    # Tambah garis-garis atau kotak-kotak
    step = np.random.randint(8, 25)
    line_color = np.clip(base_color + np.random.randint(-80, 80, 3), 0, 255).astype(np.uint8)
    for i in range(0, size, step):
        arr[i:i+2, :] = line_color
        if np.random.random() > 0.5:
            arr[:, i:i+2] = line_color
    img = Image.fromarray(arr)
    if np.random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def generate_one():
    """Pilih secara acak salah satu jenis gambar background"""
    generators = [random_solid, random_gradient, random_noise, random_shapes, random_texture]
    gen = np.random.choice(generators)
    img = gen(IMG_SIZE)
    # Random brightness adjustment (simulasi variasi cahaya)
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.5, 1.5)
        arr = np.clip(np.array(img, dtype=np.float32) * factor, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def generate_dataset(output_dir: Path, count: int, label: str):
    """Generate sejumlah gambar dan simpan ke folder"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = generate_one()
        filename = f"bg_{label}_{i:04d}.jpg"
        img.save(output_dir / filename, "JPEG", quality=92)
    print(f"   {count} gambar disimpan ke: {output_dir}")


def main(data_dir: Path):
    print("\n" + "="*60)
    print("    GENERATE DATASET 'bukan_uang' (Background)")
    print("="*60)
    print(f"  Target: {data_dir}")
    print(f"  Train: {NUM_TRAIN} gambar | Val: {NUM_VAL} | Test: {NUM_TEST}")
    print()

    generate_dataset(data_dir / "train" / "bukan_uang", NUM_TRAIN, "train")
    generate_dataset(data_dir / "val" / "bukan_uang", NUM_VAL, "val")
    generate_dataset(data_dir / "test" / "bukan_uang", NUM_TEST, "test")

    print("\n" + "="*60)
    print("   DATASET 'bukan_uang' BERHASIL DI-GENERATE!")
    print("  Sekarang jalankan train_model.py untuk melatih ulang CNN")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate background dataset")
    parser.add_argument(
        "--data_dir", type=str,
        default="dataset_split",
        help="Folder dataset_split"
    )
    args = parser.parse_args()
    main(Path(args.data_dir))
