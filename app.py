"""
====================================================================
  APLIKASI WEB DETEKSI MATA UANG RUPIAH (FLASK)
  Aksesibilitas Penyandang Tunanetra Berbasis Custom CNN (99.93% Acc)
  Versi: TFLite (ringan, untuk deployment PythonAnywhere)
====================================================================
"""

import os
import io
from pathlib import Path
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Konfigurasi Model Custom CNN (From Scratch - 8 Kelas)
BASE_DIR = Path(__file__).resolve().parent
TFLITE_PATH = BASE_DIR / "models" / "rupiah_cnn.tflite"
KERAS_PATH  = BASE_DIR / "models" / "rupiah_cnn_best.keras"

IMG_HEIGHT = 128
IMG_WIDTH  = 128

# 8 KELAS SESUAI HASIL EVALUASI MODEL CUSTOM CNN
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
    "100000": "Uang Seratus Ribu Rupiah",
    "bukan_uang": ""
}

# ── Muat interpreter TFLite ──────────────────────────────────────
USE_TFLITE = TFLITE_PATH.exists()

if USE_TFLITE:
    print(f"Memuat model Custom CNN TFLite dari: {TFLITE_PATH.name}...")
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
        interpreter = Interpreter(model_path=str(TFLITE_PATH))
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore
            interpreter = tflite.Interpreter(model_path=str(TFLITE_PATH))
        except ImportError:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))

    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    num_out_classes = output_details[0]['shape'][-1]
    if num_out_classes == 9 and "bukan_uang" not in CLASSES:
        CLASSES.append("bukan_uang")
    print(f"Model Custom CNN TFLite berhasil dimuat ({num_out_classes} kelas)!")

else:
    print(f"TFLite tidak ditemukan, memuat Custom CNN Keras: {KERAS_PATH.name}...")
    import tensorflow as tf
    model = tf.keras.models.load_model(KERAS_PATH)
    num_out_classes = model.output_shape[-1]
    if num_out_classes == 9 and "bukan_uang" not in CLASSES:
        CLASSES.append("bukan_uang")
    print(f"Model Custom CNN Keras berhasil dimuat ({num_out_classes} kelas)!")


def preprocess_image(pil_img):
    """Samakan metode resize NEAREST dan normalisasi 1/255 dengan ImageDataGenerator Keras"""
    img_resized = pil_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.NEAREST)
    img_array   = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0), img_array


def predict_probs(pil_img):
    """Mengembalikan array probabilitas prediksi dari TFLite atau Keras"""
    batch_array, _ = preprocess_image(pil_img)
    if USE_TFLITE:
        interpreter.set_tensor(input_details[0]['index'], batch_array)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]
    else:
        return model.predict(batch_array, verbose=0)[0]


def predict_crop(pil_img):
    """Memprediksi satu potongan gambar dengan Multi-Crop (Full + Center Zoom) & Filter Entropi"""
    # 1. Filter kontras: gambar terlalu polos (tembok/meja/gelap) -> bukan uang
    gray     = pil_img.convert("L")
    arr_gray = np.array(gray, dtype=np.float32)
    if arr_gray.std() < 18.0:
        return "bukan_uang", 99.0

    # 2. Ambil prediksi gambar utuh dan potongan tengah (untuk membuang background wajah/tangan di pinggir kotak)
    w, h = pil_img.size
    center_img = pil_img.crop((int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88)))

    preds_full   = predict_probs(pil_img)
    preds_center = predict_probs(center_img)

    # Rata-ratakan hasil prediksi dengan bobot lebih besar pada area tengah uang
    preds = (preds_full * 0.4) + (preds_center * 0.6)

    idx  = int(np.argmax(preds))
    conf = float(preds[idx]) * 100
    nom_str = CLASSES[idx]

    # 3. Karena model 8 kelas tidak punya kelas 'bukan_uang', gunakan syarat confidence >= 85%
    # dan pastikan selisih dengan juara ke-2 cukup jauh (tidak menebak acak)
    sorted_preds = np.sort(preds)[::-1]
    margin = float(sorted_preds[0] - sorted_preds[1]) * 100

    if conf < 82.0 or margin < 35.0:
        return "bukan_uang", conf

    return nom_str, conf


def format_nominal_speech(value):
    """Mengubah nominal angka menjadi pengucapan bahasa Indonesia"""
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

    mode = request.form.get("mode", "single")

    try:
        image_bytes = file.read()
        full_img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h        = full_img.size

        # 1. Prediksi citra penuh
        nominal_full, conf_full = predict_crop(full_img)

        # 2. Cek 2 lembar berdampingan (HANYA untuk gambar yang sangat lebar)
        is_multi     = False
        aspect_ratio = w / max(h, 1)

        if aspect_ratio >= 1.6 and w >= 200:
            left_crop  = full_img.crop((0, 0, int(w * 0.52), h))
            right_crop = full_img.crop((int(w * 0.48), 0, w, h))

            nom_left,  conf_left  = predict_crop(left_crop)
            nom_right, conf_right = predict_crop(right_crop)

            is_valid_money_both = (
                nom_left != "bukan_uang" and nom_right != "bukan_uang"
                and conf_left >= 82.0 and conf_right >= 82.0
            )

            if is_valid_money_both:
                is_distinct_dual = (
                    nom_left != nom_right and (
                        nominal_full == "bukan_uang" or conf_full < 75.0
                        or nom_left == nominal_full or nom_right == nominal_full
                    )
                )
                is_same_dual = (
                    mode == "dual" and nom_left == nom_right and aspect_ratio >= 2.6
                )

                if is_distinct_dual or is_same_dual:
                    is_multi  = True
                    val_left  = int(nom_left)
                    val_right = int(nom_right)
                    total_val = val_left + val_right
                    avg_conf  = round((conf_left + conf_right) / 2, 2)

                    suara_kiri  = SPEECH_TEXTS.get(nom_left,  f"Rp {val_left}")
                    suara_kanan = SPEECH_TEXTS.get(nom_right, f"Rp {val_right}")
                    total_speech = format_nominal_speech(total_val)

                    return jsonify({
                        "success": True,
                        "is_multi": True,
                        "is_money": True,
                        "val_left": val_left,
                        "val_right": val_right,
                        "nominal_left":  f"Rp {val_left:,}".replace(",", "."),
                        "nominal_right": f"Rp {val_right:,}".replace(",", "."),
                        "nominal":       f"Rp {total_val:,}".replace(",", "."),
                        "numeric_value": total_val,
                        "confidence":    avg_conf,
                        "speech_text":   f"Dua lembar terdeteksi: {suara_kiri} dan {suara_kanan}. Totalnya {total_speech}.",
                        "detail_text":   f"Lembar 1: Rp {val_left:,} ({round(conf_left)}%) | Lembar 2: Rp {val_right:,} ({round(conf_right)}%)".replace(",", ".")
                    })

        # 3. Lembaran Tunggal
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

        val_single        = int(nominal_full)
        nominal_formatted = f"Rp {val_single:,}".replace(",", ".")
        suara_teks        = SPEECH_TEXTS.get(nominal_full, f"Uang {nominal_full} Rupiah")

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