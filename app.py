"""
Flask App - Fish Classification
Web app untuk klasifikasi jenis ikan menggunakan MobileNetV2 atau EfficientNetB0
(toggle antar model tersedia di UI).

Jalankan dari root project:
    python app.py
"""

import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # maks 8MB per upload

# ====== LOAD MODEL & CLASS MAPPING SAAT STARTUP ======
print("Memuat model MobileNetV2...")
mobilenet_model = load_model(MODELS_DIR / "mobilenet_best.h5", compile=False)

print("Memuat model EfficientNetB0...")
efficientnet_model = load_model(MODELS_DIR / "efficientnet_best.h5", compile=False)

with open(MODELS_DIR / "class_indices.json") as f:
    class_indices = json.load(f)
# class_indices: {"NamaKelas": index} -> kita butuh kebalikannya: {index: "NamaKelas"}
idx_to_class = {v: k for k, v in class_indices.items()}

MODELS = {
    "mobilenet": mobilenet_model,
    "efficientnet": efficientnet_model,
}

# ====== FAKTA SINGKAT TIAP SPESIES (ditampilkan di Fish Info Card) ======
FISH_INFO = {
    "Black Sea Sprat": {
        "nama_latin": "Clupeonella cultriventris",
        "habitat": "Laut Hitam & perairan payau sekitarnya",
        "fakta": "Ikan kecil bertubuh ramping yang hidup bergerombol besar dan jadi sumber pakan penting bagi ikan predator serta burung laut.",
    },
    "Gilt-Head Bream": {
        "nama_latin": "Sparus aurata",
        "habitat": "Laut Mediterania & Samudra Atlantik Timur",
        "fakta": "Punya garis emas khas di antara kedua matanya, dan merupakan salah satu ikan budidaya paling populer di Eropa selatan.",
    },
    "Hourse Mackerel": {
        "nama_latin": "Trachurus trachurus",
        "habitat": "Samudra Atlantik Timur & Laut Mediterania",
        "fakta": "Punya deretan sisik keras (scute) di sepanjang garis rusuknya yang berfungsi sebagai pelindung tubuh.",
    },
    "Red Mullet": {
        "nama_latin": "Mullus barbatus",
        "habitat": "Laut Mediterania & Atlantik Timur, dasar berlumpur/berpasir",
        "fakta": "Punya sepasang 'kumis' (barbel) di dagu yang dipakai untuk meraba dan mencari makanan di dasar laut.",
    },
    "Red Sea Bream": {
        "nama_latin": "Pagellus bogaraveo",
        "habitat": "Samudra Atlantik Timur & Laut Mediterania, perairan agak dalam",
        "fakta": "Punya bintik hitam khas di dekat insang yang jadi ciri pembeda utamanya dari spesies bream lain.",
    },
    "Sea Bass": {
        "nama_latin": "Dicentrarchus labrax",
        "habitat": "Pesisir Atlantik Timur & Laut Mediterania",
        "fakta": "Predator agresif yang jadi incaran utama pemancing rekreasi, dan salah satu ikan konsumsi paling bernilai di Eropa.",
    },
    "Shrimp": {
        "nama_latin": "Bervariasi tergantung spesies",
        "habitat": "Perairan laut dangkal hingga payau di seluruh dunia",
        "fakta": "Meski disertakan dalam dataset ikan, udang sebenarnya krustasea (bukan ikan) yang punya eksoskeleton dan berjalan pakai kaki.",
    },
    "Striped Red Mullet": {
        "nama_latin": "Mullus surmuletus",
        "habitat": "Laut Mediterania & Atlantik Timur, dasar berbatu/berpasir",
        "fakta": "Mirip Red Mullet tapi punya garis-garis kuning memanjang di sisi tubuhnya, dan warnanya berubah cepat setelah ditangkap.",
    },
    "Trout": {
        "nama_latin": "Oncorhynchus mykiss / Salmo trutta (tergantung jenis)",
        "habitat": "Sungai, danau air tawar dingin, dan sebagian hidup di laut",
        "fakta": "Sangat sensitif terhadap kualitas air, sehingga sering dipakai sebagai indikator kesehatan ekosistem sungai.",
    },
}

print("Semua model siap. Server Flask jalan.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path: str, model_name: str) -> np.ndarray:
    """Preprocessing gambar sesuai model yang dipilih.

    PENTING: EfficientNetB0 punya layer Rescaling/Normalization internal
    di dalam arsitekturnya (expect piksel mentah 0-255), sedangkan
    MobileNetV2 tidak -- jadi perlu rescale manual (bagi 255) khusus
    untuk MobileNetV2 saja. Ini harus konsisten dengan preprocessing
    saat training di src/train.py.
    """
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if model_name == "mobilenet":
        img_array = img_array / 255.0

    return img_array


@app.route("/")
def index():
    class_names = sorted(class_indices, key=lambda k: class_indices[k])
    return render_template("index.html", class_names=class_names)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    file = request.files["file"]
    model_name = request.form.get("model", "efficientnet")

    if model_name not in MODELS:
        return jsonify({"error": "Model tidak dikenali"}), 400

    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung. Gunakan PNG, JPG, atau JPEG"}), 400

    # Simpan file sementara
    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)

    try:
        img_array = preprocess_image(str(save_path), model_name)
        model = MODELS[model_name]
        predictions = model.predict(img_array, verbose=0)[0]

        # Ambil top-3 prediksi
        top_indices = predictions.argsort()[-3:][::-1]
        top_predictions = [
            {
                "class_name": idx_to_class[int(idx)],
                "confidence": float(predictions[idx]) * 100,
            }
            for idx in top_indices
        ]

        predicted_name = top_predictions[0]["class_name"]
        result = {
            "predicted_class": predicted_name,
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
            "model_used": model_name,
            "image_url": f"/static/uploads/{file.filename}",
            "fish_info": FISH_INFO.get(predicted_name),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan saat memproses gambar: {str(e)}"}), 500


@app.route("/predict_compare", methods=["POST"])
def predict_compare():
    """Jalankan prediksi pakai KEDUA model sekaligus untuk gambar yang sama,
    dipakai oleh fitur 'Bandingkan Kedua Model' di frontend."""
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung. Gunakan PNG, JPG, atau JPEG"}), 400

    save_path = UPLOAD_DIR / file.filename
    file.save(save_path)

    try:
        results = {}
        for model_name, model in MODELS.items():
            img_array = preprocess_image(str(save_path), model_name)
            predictions = model.predict(img_array, verbose=0)[0]
            top_indices = predictions.argsort()[-3:][::-1]
            top_predictions = [
                {
                    "class_name": idx_to_class[int(idx)],
                    "confidence": float(predictions[idx]) * 100,
                }
                for idx in top_indices
            ]
            predicted_name = top_predictions[0]["class_name"]
            results[model_name] = {
                "predicted_class": predicted_name,
                "confidence": top_predictions[0]["confidence"],
                "top_predictions": top_predictions,
                "fish_info": FISH_INFO.get(predicted_name),
            }

        agree = results["mobilenet"]["predicted_class"] == results["efficientnet"]["predicted_class"]

        return jsonify({
            "image_url": f"/static/uploads/{file.filename}",
            "results": results,
            "agree": agree,
        })

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan saat memproses gambar: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)