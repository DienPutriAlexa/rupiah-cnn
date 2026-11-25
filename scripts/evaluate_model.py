"""
====================================================================
  LANGKAH 7: Pengujian Model pada Test Set & Matriks Evaluasi
  Deteksi Mata Uang Rupiah - CNN Aksesibilitas
====================================================================

Fungsi:
  - Menguji model rupiah_cnn_best.keras pada 2.700 gambar test set
  - Menghitung Confusion Matrix (Visual Heatmap)
  - Menghitung Precision, Recall, F1-Score per kelas nominal
  - Menyimpan artefak matriks ke CSV, TXT, dan gambar PNG

Cara Pakai:
  python evaluate_model.py
"""

import os
import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG_HEIGHT = 128
IMG_WIDTH  = 128
BATCH_SIZE = 32

def evaluate(model_path: Path, test_dir: Path, output_dir: Path):
    print("\n" + "="*60)
    print("   PENGUJIAN MODEL PADA DATASET TEST (Matriks Evaluasi)")
    print("="*60)
    print(f"Model File : {model_path}")
    print(f"Test Dir   : {test_dir}\n")

    if not model_path.exists():
        print(f"[ERROR] File model '{model_path}' tidak ditemukan!")
        return

    # Load Model
    print("Memuat model...")
    model = tf.keras.models.load_model(model_path)
    is_mobilenet = "mobilenet" in getattr(model, "name", "").lower()
    print(f"Model termuat: {model.name} (MobileNetV2: {is_mobilenet})")

    # Load Test Generator (tanpa shuffle agar urutan label cocok)
    # MobileNetV2 memiliki layer preprocess_input internal sehingga tidak perlu rescale
    if is_mobilenet:
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    else:
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    class_labels = list(test_generator.class_indices.keys())
    display_labels = []
    for cls in class_labels:
        if cls == "bukan_uang":
            display_labels.append("Bukan Uang")
        else:
            display_labels.append(f"Rp {int(cls):,}".replace(",", "."))

    # Prediksi
    print("\nMelakukan prediksi pada seluruh data uji (2.700 gambar)...")
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # 1. Classification Report (Precision, Recall, F1)
    print("\n" + "="*60)
    print("   CLASSIFICATION REPORT (Precision, Recall, F1-Score)")
    print("="*60)
    report_dict = classification_report(
        y_true, y_pred, target_names=display_labels, output_dict=True
    )
    report_text = classification_report(
        y_true, y_pred, target_names=display_labels, digits=4
    )
    print(report_text)

    # Simpan report text
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "laporan_matriks_evaluasi.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("  MATRIKS PENGUJIAN MODEL CNN - DETEKSI MATA UANG RUPIAH\n")
        f.write("="*60 + "\n\n")
        f.write(report_text)
    print(f" Laporan teks disimpan: {report_file}")

    # Simpan report CSV
    csv_file = output_dir / "matriks_evaluasi.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nominal", "Precision", "Recall", "F1-Score", "Jumlah Data Uji (Support)"])
        for lbl in display_labels:
            m = report_dict[lbl]
            writer.writerow([lbl, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1-score']:.4f}", m['support']])
        acc = report_dict['accuracy']
        writer.writerow(["Overall Accuracy", f"{acc:.4f}", "-", "-", len(y_true)])
    print(f" CSV matriks disimpan : {csv_file}")

    # 2. Confusion Matrix Visual (Heatmap)
    print("\nMembuat visualisasi Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True
    )
    plt.title('Confusion Matrix - Deteksi Mata Uang Rupiah (CNN)', fontsize=14, pad=15)
    plt.xlabel('Prediksi Model', fontsize=12)
    plt.ylabel('Label Sebenarnya (Ground Truth)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  Gambar Confusion Matrix disimpan: {cm_path}")

    print("\n" + "="*60)
    print("   PENGUJIAN SELESAI DENGAN SUKSES!")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi Model CNN")
    default_model = "models/rupiah_cnn_best.keras"
    parser.add_argument("--model_path", type=str, default=default_model)
    parser.add_argument("--test_dir", type=str, default="dataset_split/test")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    evaluate(Path(args.model_path), Path(args.test_dir), Path(args.output_dir))