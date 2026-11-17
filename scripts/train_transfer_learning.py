"""
====================================================================
  LANGKAH 6B: Training Transfer Learning Model (MobileNetV2)
  Deteksi Mata Uang Rupiah - CNN Aksesibilitas Tunanetra
  Optimasi Akurasi & Waktu Komputasi Menggunakan Pre-trained Weights
====================================================================

Fitur:
  - Backbone: MobileNetV2 pre-trained pada ImageNet (Weights Frozen)
  - Custom Classification Head khusus 8 Kelas Rupiah
  - Strategi Dua Tahap:
      Tahap 1: Feature Extraction (Bobot base dibekukan, latih classifier head)
      Tahap 2: Fine-Tuning (Buka 30 layer teratas base model, LR sangat kecil)
  - Data Augmentation & Preprocessing resmi MobileNetV2
  - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
  - Ekspor grafik akurasi & loss untuk laporan perbandingan tugas besar

Cara Pakai:
  python train_transfer_learning.py --data_dir "D:/path/ke/rupiah-cnn/dataset_split"
"""

import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ─── KONFIGURASI PARAMETER ─────────────────────────────────────────
IMG_HEIGHT       = 128
IMG_WIDTH        = 128
BATCH_SIZE       = 32
INITIAL_EPOCHS   = 15   # Epoch untuk Feature Extraction
FINE_TUNE_EPOCHS = 10   # Epoch tambahan untuk Fine-Tuning
TOTAL_EPOCHS     = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
NUM_CLASSES      = 8
# ───────────────────────────────────────────────────────────────────

def build_transfer_learning_model(input_shape=(128, 128, 3), num_classes=8):
    """
    Membangun model Transfer Learning dengan backbone MobileNetV2.
    MobileNetV2 dipilih karena sangat efisien (arsitektur Inverted Residuals &
    Depthwise Separable Convolutions), ringan untuk CPU/PythonAnywhere dan mobile.
    """
    # 1. Base Pre-trained Model (tanpa fully connected classifier asli ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # 2. Pembekuan Bobot (Freezing) untuk Fase Feature Extraction
    base_model.trainable = False

    # 3. Rakit Arsitektur Lengkap
    inputs = layers.Input(shape=input_shape)
    
    # Preprocessing khusus MobileNetV2 (skala [-1, 1])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Lewatkan ke base model (training=False agar BatchNormalization tetap dalam mode inferensi)
    x = base_model(x, training=False)
    
    # Custom Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Rupiah_MobileNetV2_TransferLearning")
    return model, base_model

def plot_combined_history(history_initial, history_fine, save_dir: Path):
    """Menyimpan grafik 2 fase (Feature Extraction + Fine-Tuning)"""
    acc = history_initial.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history_initial.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history_initial.history['loss'] + history_fine.history['loss']
    val_loss = history_initial.history['val_loss'] + history_fine.history['val_loss']

    epochs_range = range(1, len(acc) + 1)
    split_epoch = len(history_initial.history['accuracy'])

    plt.figure(figsize=(14, 5))

    # Plot Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2, color='#1e3a8a')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2, linestyle='--', color='#0ea5e9')
    plt.axvline(x=split_epoch, color='red', linestyle=':', label='Mulai Fine-Tuning')
    plt.title('Akurasi Transfer Learning (Feature Extraction & Fine-Tuning)', fontsize=11)
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', linewidth=2, color='#b91c1c')
    plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2, linestyle='--', color='#f97316')
    plt.axvline(x=split_epoch, color='red', linestyle=':', label='Mulai Fine-Tuning')
    plt.title('Loss Transfer Learning per Epoch', fontsize=11)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    save_path = save_dir / "grafik_transfer_learning_mobilenetv2.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Grafik training Transfer Learning disimpan di: {save_path}")

def train(data_dir: Path, output_dir: Path):
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  🚀 PIPELINE DATA AUGMENTATION (TRANSFER LEARNING)")
    print("="*65)

    # Catatan: Normalisasi dilakukan oleh preprocess_input di dalam layer model
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

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

    print(f"\nIndex Kelas Rupiah ({len(train_generator.class_indices)} Kelas):")
    for cls_name, idx in train_generator.class_indices.items():
        print(f"  [{idx}] -> Rp {cls_name}")

    # ─────────────────────────────────────────────────────────────
    # FASE 1: FEATURE EXTRACTION
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  🧠 [FASE 1] MEMBANGUN MODEL: FEATURE EXTRACTION (FREEZING)")
    print("="*65)
    model, base_model = build_transfer_learning_model((IMG_HEIGHT, IMG_WIDTH, 3), NUM_CLASSES)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    best_model_path = models_dir / "rupiah_mobilenetv2_best.keras"
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

    print("\n" + "="*65)
    print("  🏋️ [FASE 1] TRAINING CLASSIFICATION HEAD (15 Epochs)")
    print("="*65)
    history_initial = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,
        validation_data=val_generator,
        callbacks=my_callbacks
    )

    # ─────────────────────────────────────────────────────────────
    # FASE 2: FINE-TUNING
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  🔥 [FASE 2] UNFREEZING TOP LAYERS UNTUK FINE-TUNING")
    print("="*65)
    # Buka unfreeze pada 30 layer teratas MobileNetV2
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Gunakan Learning Rate yang SANGAT KECIL (1e-5) agar bobot ImageNet tidak rusak
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Total layer di backbone: {len(base_model.layers)}")
    print(f"Layer yang di-unfreeze : {len(base_model.layers) - fine_tune_at} layer teratas")

    print("\n" + "="*65)
    print("  🏋️ [FASE 2] TRAINING FINE-TUNING (10 Epochs Tambahan)")
    print("="*65)
    history_fine = model.fit(
        train_generator,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history_initial.epoch[-1] + 1,
        validation_data=val_generator,
        callbacks=my_callbacks
    )

    # Simpan grafik gabungan
    plot_combined_history(history_initial, history_fine, models_dir)
    print("\n" + "="*65)
    print("  ✅ PIPELINE TRANSFER LEARNING LENGKAP SELESAI!")
    print(f"  Model bobot terbaik tersimpan di: {best_model_path}")
    print("="*65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Transfer Learning MobileNetV2 Rupiah")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder dataset_split")
    parser.add_argument("--output_dir", type=str, default=".", help="Folder output untuk model & grafik")
    args = parser.parse_args()

    train(Path(args.data_dir), Path(args.output_dir))
