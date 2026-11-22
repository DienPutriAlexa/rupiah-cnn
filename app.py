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
MOBILENET_PATH = BASE_DIR / "models" / "rupiah_mobilenetv2_best.keras"
CUSTOM_PATH    = BASE_DIR / "models" / "rupiah_cnn_best.keras"

# Prioritaskan model Transfer Learning jika sudah di-train, fallback ke Custom CNN
MODEL_PATH = MOBILENET_PATH if MOBILENET_PATH.exists() else CUSTOM_PATH
IMG_HEIGHT = 128
IMG_WIDTH  = 128

# URUTAN KELAS PERSIS SESUAI INDEKS KERAS (ALFABETIS FOLDER)
# 0: 1000, 1: 10000, 2: 100000, 3: 2000, 4: 20000, 5: 5000, 6: 50000, 7: 75000, 8: bukan_uang
CLASSES = [
    "1000",
    "10000",
    "100000",
    "2000",
    "20000",
    "5000",
    "50000",
    "75000",
    "bukan_uang"
]

SPEECH_TEXTS = {
    "1000": "Uang Seribu Rupiah",
    "2000": "Uang Dua Ribu Rupiah",
    "5000": "Uang Lima Ribu Rupiah",
    "10000": "Uang Sepuluh Ribu Rupiah",
    "20000": "Uang Dua Puluh Ribu Rupiah",
    "50000": "Uang Lima Puluh Ribu Rupiah",
    "75000": "Uang Tujuh Puluh Lima Ribu Rupiah",
    "100000": "Uang Seratus Ribu Rupiah",
    "bukan_uang": ""
}

print(f"Memuat model dari: {MODEL_PATH.name}...")
model = tf.keras.models.load_model(MODEL_PATH)
is_mobilenet = "mobilenet" in getattr(model, "name", "").lower()
print(f"Model berhasil dimuat! (Tipe: {'Transfer Learning MobileNetV2' if is_mobilenet else 'Custom CNN'})")

def predict_crop(pil_img):
    """Memprediksi satu potongan gambar menggunakan model CNN"""
    # 1. Cek apakah citra terlalu polos / gelap / tidak ada objek (tembok, meja polos, blur)
    gray = pil_img.convert("L")
    arr_gray = np.array(gray, dtype=np.float32)
    # Jika kontras terlalu rendah (std dev < 15), hampir pasti bukan lembaran uang bertekstur
    if arr_gray.std() < 15.0:
        return "bukan_uang", 99.0

    img_resized = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Custom CNN butuh normalisasi 0..1 manual,
    # sedangkan MobileNetV2 sudah memiliki layer preprocess_input internal
    if not is_mobilenet:
        img_array = img_array / 255.0
        
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx]) * 100
    nom_str = CLASSES[idx]

    # Jika keyakinan model rendah (< 70%), anggap bukan uang
    if conf < 70.0:
        return "bukan_uang", conf

    return nom_str, conf

def format_nominal_speech(value):
    """Mengubah nominal angka menjadi pengucapan audio bahasa Indonesia yang fasih"""
    bilang = {
        1000: "seribu rupiah", 2000: "dua ribu rupiah", 3000: "tiga ribu rupiah",
        4000: "empat ribu rupiah", 5000: "lima ribu rupiah", 6000: "enam ribu rupiah",
        7000: "tujuh ribu rupiah", 8000: "delapan ribu rupiah", 9000: "sembilan ribu rupiah",
        10000: "sepuluh ribu rupiah", 11000: "sebelas ribu rupiah", 12000: "dua belas ribu rupiah",
        15000: "lima belas ribu rupiah", 16000: "enam belas ribu rupiah", 17000: "tujuh belas ribu rupiah",
        20000: "dua puluh ribu rupiah", 22000: "dua puluh dua ribu rupiah", 25000: "dua puluh lima ribu rupiah",
        30000: "tiga puluh ribu rupiah", 35000: "tiga puluh lima ribu rupiah", 40000: "empat puluh ribu rupiah",
        50000: "lima puluh ribu rupiah", 55000: "lima puluh lima ribu rupiah", 60000: "enam puluh ribu rupiah",
        70000: "tujuh puluh ribu rupiah", 75000: "tujuh puluh lima ribu rupiah", 80000: "delapan puluh ribu rupiah",
        85000: "delapan puluh lima ribu rupiah", 90000: "sembilan puluh ribu rupiah", 95000: "sembilan puluh lima ribu rupiah",
        100000: "seratus ribu rupiah", 105000: "seratus lima ribu rupiah", 110000: "seratus sepuluh ribu rupiah",
        120000: "seratus dua puluh ribu rupiah", 125000: "seratus dua puluh lima ribu rupiah",
        150000: "seratus lima puluh ribu rupiah", 175000: "seratus tujuh puluh lima ribu rupiah",
        200000: "dua ratus ribu rupiah"
    }
    return bilang.get(value, f"Rp {value:,}".replace(",", ".") + " rupiah")

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

    mode = request.form.get("mode", "single")  # "single" atau "dual"

    try:
        image_bytes = file.read()
        full_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = full_img.size

        # 1. Prediksi citra penuh (Full Image)
        nominal_full, conf_full = predict_crop(full_img)

        # 2. Cek apakah ada 2 lembar uang berdampingan (Side-by-Side Banknotes)
        # Kondisi agar tidak memotong 1 lembar uang utuh menjadi 2 potongan palsu:
        # - Rasio lebar terhadap tinggi harus sangat lebar (aspect_ratio >= 1.6)
        # - Potongan kiri dan kanan harus sama-sama bernilai uang valid dengan akurasi >= 80%
        # - Salah satu potongan harus cocok dengan nominal_full, ATAU nominal_full ragu-ragu (< 70%) / bukan_uang
        is_multi = False
        aspect_ratio = w / max(h, 1)

        if aspect_ratio >= 1.6 and w >= 200:
            left_crop = full_img.crop((0, 0, int(w * 0.52), h))
            right_crop = full_img.crop((int(w * 0.48), 0, w, h))
            
            nom_left, conf_left = predict_crop(left_crop)
            nom_right, conf_right = predict_crop(right_crop)
            
            is_valid_money_both = (
                nom_left != "bukan_uang" and nom_right != "bukan_uang"
                and conf_left >= 80.0 and conf_right >= 80.0
            )

            if is_valid_money_both:
                # Jika beda nominal (misal 5.000 dan 10.000 berdampingan)
                is_distinct_dual = (
                    nom_left != nom_right and (
                        nominal_full == "bukan_uang" or conf_full < 70.0
                        or nom_left == nominal_full or nom_right == nominal_full
                    )
                )
                # Jika pecahan sama berdampingan khusus saat mode dual dan citra sangat lebar
                is_same_dual = (
                    mode == "dual" and nom_left == nom_right and aspect_ratio >= 2.6
                )

                if is_distinct_dual or is_same_dual:
                    is_multi = True
                    val_left = int(nom_left)
                    val_right = int(nom_right)
                    total_val = val_left + val_right
                    avg_conf = round((conf_left + conf_right) / 2, 2)
                    
                    suara_kiri = SPEECH_TEXTS.get(nom_left, f"Rp {val_left}")
                    suara_kanan = SPEECH_TEXTS.get(nom_right, f"Rp {val_right}")
                    total_speech = format_nominal_speech(total_val)
                    
                    return jsonify({
                        "success": True,
                        "is_multi": True,
                        "is_money": True,
                        "val_left": val_left,
                        "val_right": val_right,
                        "nominal_left": f"Rp {val_left:,}".replace(",", "."),
                        "nominal_right": f"Rp {val_right:,}".replace(",", "."),
                        "nominal": f"Rp {total_val:,}".replace(",", "."),
                        "numeric_value": total_val,
                        "confidence": avg_conf,
                        "speech_text": f"Dua lembar terdeteksi: {suara_kiri} dan {suara_kanan}. Totalnya {total_speech}.",
                        "detail_text": f"Lembar 1: Rp {val_left:,} ({round(conf_left)}%) | Lembar 2: Rp {val_right:,} ({round(conf_right)}%)".replace(",", ".")
                    })

        # 3. Prediksi Lembaran Tunggal (Single Banknote)
        if nominal_full == "bukan_uang":
            return jsonify({
                "success": True,
                "is_multi": False,
                "nominal": "-",
                "raw_nominal": "bukan_uang",
                "numeric_value": 0,
                "confidence": round(conf_full, 2),
                "speech_text": "",
                "speech_nominal": "",
                "is_money": False
            })

        val_single = int(nominal_full)
        nominal_formatted = f"Rp {val_single:,}".replace(",", ".")
        suara_teks = SPEECH_TEXTS.get(nominal_full, f"Uang {nominal_full} Rupiah")

        return jsonify({
            "success": True,
            "is_multi": False,
            "nominal": nominal_formatted,
            "raw_nominal": nominal_full,
            "numeric_value": val_single,
            "confidence": round(conf_full, 2),
            "speech_text": f"{suara_teks}.",
            "speech_nominal": suara_teks,
            "is_money": True
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
