
import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cottoncare_mobilenet.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "class_names.json")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "CottonCare AI backend is running",
        "classes": class_names
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image"}), 400

    file = request.files["image"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((160, 160))

        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
        index = int(np.argmax(predictions))

        return jsonify({
            "disease": class_names[index],
            "confidence": round(float(predictions[index]) * 100, 2),
            "message": "AI prediction only; verify with a local agriculture expert."
        })
    except Exception as error:
        return jsonify({"error": "Could not process image"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
