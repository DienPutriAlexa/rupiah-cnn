"""
====================================================================
  LANGKAH 6: Training Custom CNN Model Dari Nol (From Scratch)
  Deteksi Mata Uang Rupiah - CNN Aksesibilitas Tunanetra
====================================================================

Fitur:
  - Arsitektur Custom CNN (Memenuhi Poin 7 Tugas Besar)
  - Data Augmentation (simulasi posisi uang di tangan tunanetra)
  - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
  - Ekspor grafik akurasi & loss untuk Laporan & PPT

Cara Pakai:
  python train_model.py --data_dir "D:/path/ke/rupiah-cnn/dataset_split"
"""

import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ─── KONFIGURASI PARAMETER ─────────────────────────────────────────
IMG_HEIGHT  = 128
IMG_WIDTH   = 128
BATCH_SIZE  = 32
EPOCHS      = 25
NUM_CLASSES = 8
# ───────────────────────────────────────────────────────────────────

def build_custom_cnn(input_shape=(128, 128, 3), num_classes=8):
    """
    Arsitektur Custom CNN 4-Blok Konvolusi yang dirancang efisien
    khusus untuk klasifikasi Rupiah pada CPU/Cloud Server (PythonAnywhere).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Blok 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Blok 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Blok 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Blok 4
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classification Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_training_history(history, save_dir: Path):
    """Menyimpan grafik akurasi dan loss untuk laporan tugas besar"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2, linestyle='--')
    plt.title('Akurasi Model per Epoch', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2, linestyle='--')
    plt.title('Loss Model per Epoch', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    save_path = save_dir / "grafik_training_cnn.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Grafik training disimpan di: {save_path}")

def train(data_dir: Path, output_dir: Path):
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  🚀 MEMPERSIAPKAN DATA & PIPELINE TRAINING")
    print("="*60)

    # Data Augmentation & Normalisasi
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print(f"\nIndex Kelas Model:")
    for cls_name, idx in train_generator.class_indices.items():
        print(f"  [{idx}] -> Rp {cls_name}")

    # Build Model
    print("\n" + "="*60)
    print("  🧠 MEMBANGUN ARSITEKTUR CUSTOM CNN")
    print("="*60)
    model = build_custom_cnn((IMG_HEIGHT, IMG_WIDTH, 3), NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_loss' if hasattr(tf.keras.losses, 'categorical_loss') else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    best_model_path = models_dir / "rupiah_cnn_best.keras"
    my_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n" + "="*60)
    print("  🏋️ MEMULAI PROSES TRAINING (Custom CNN)")
    print("="*60)
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=my_callbacks
    )

    # Simpan grafik & log
    plot_training_history(history, models_dir)
    print("\n" + "="*60)
    print(f"  ✅ TRAINING SELESAI!")
    print(f"  Model terbaik disimpan di: {best_model_path}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Custom CNN Rupiah")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder dataset_split")
    parser.add_argument("--output_dir", type=str, default=".", help="Folder output untuk model & grafik")
    args = parser.parse_args()

    train(Path(args.data_dir), Path(args.output_dir))
