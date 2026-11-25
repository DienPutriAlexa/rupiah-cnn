"""
Script konversi model Custom CNN Keras -> TFLite
Jalankan di PC lokal: python scripts/convert_to_tflite.py
"""
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rupiah_cnn_best.keras"
OUTPUT_PATH = BASE_DIR / "models" / "rupiah_cnn.tflite"

print(f"Memuat model Custom CNN: {MODEL_PATH.name}...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Mengkonversi ke TFLite (Float16 agar akurasi tetap tajam)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

OUTPUT_PATH.write_bytes(tflite_model)
size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
print(f"Berhasil! Disimpan ke: {OUTPUT_PATH.name} ({size_mb:.2f} MB)")