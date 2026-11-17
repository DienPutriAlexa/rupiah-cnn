"""
====================================================================
  APLIKASI WEB DETEKSI MATA UANG RUPIAH (FLASK)
  Aksesibilitas Penyandang Tunanetra Berbasis CNN
====================================================================
"""

import os
import io
from pathlib import Path
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

# Konfigurasi Model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rupiah_cnn_best.keras"
IMG_HEIGHT = 128
IMG_WIDTH  = 128

# URUTAN KELAS PERSIS SESUAI INDEKS KERAS (ALFABETIS FOLDER)
# 0: 1000, 1: 10000, 2: 100000, 3: 2000, 4: 20000, 5: 5000, 6: 50000, 7: 75000
CLASSES = [
    "1000",
    "10000",
    "100000",
    "2000",
    "20000",
    "5000",
    "50000",
    "75000"
]

SPEECH_TEXTS = {
    "1000": "Uang Seribu Rupiah",
    "2000": "Uang Dua Ribu Rupiah",
    "5000": "Uang Lima Ribu Rupiah",
    "10000": "Uang Sepuluh Ribu Rupiah",
    "20000": "Uang Dua Puluh Ribu Rupiah",
    "50000": "Uang Lima Puluh Ribu Rupiah",
    "75000": "Uang Tujuh Puluh Lima Ribu Rupiah",
    "100000": "Uang Seratus Ribu Rupiah"
}

print("Memuat model CNN...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model berhasil dimuat!")

def preprocess_image(image_bytes):
    """Mengubah bytes gambar menjadi tensor input CNN yang dinormalisasi"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "Tidak ada file gambar yang dikirim"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "File tidak dipilih"}), 400

    try:
        image_bytes = file.read()
        processed_img = preprocess_image(image_bytes)

        # Prediksi model CNN
        predictions = model.predict(processed_img, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx]) * 100

        nominal_str = CLASSES[predicted_idx]
        nominal_formatted = f"Rp {int(nominal_str):,}".replace(",", ".")
        suara_teks = SPEECH_TEXTS.get(nominal_str, f"Uang {nominal_str} Rupiah")

        return jsonify({
            "success": True,
            "nominal": nominal_formatted,
            "raw_nominal": nominal_str,
            "confidence": round(confidence, 2),
            "speech_text": f"{suara_teks}. Tingkat keyakinan {round(confidence)} persen."
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
