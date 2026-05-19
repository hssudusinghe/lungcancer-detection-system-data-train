from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingModuleSource]
from flask_cors import CORS  # import CORS
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)  # enable CORS for all routes
model = load_model("lung_cancer_model.h5")


class_names = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma"
]

# ================= CHECK IF IMAGE LOOKS LIKE CT SCAN =================
def is_ct_scan(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)

    # Convert to grayscale difference check
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    # CT scans are usually grayscale (R,G,B nearly equal)
    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))

    # If differences are very small → likely grayscale scan
    if diff_rg < 10 and diff_rb < 10:
        return True

    return False


# ================= PREPARE IMAGE =================
def prepare_image(image):
    img = Image.open(image).resize((224, 224))
    img = img.convert("RGB")

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    try:

        file = request.files.get("image")
        user_id = request.form.get("user_id")

        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        if not user_id:
            return jsonify({"error": "User ID missing"}), 400

        user_id = int(user_id)

        # Validate image
        if not is_ct_scan(file):
            return jsonify({
                "error": "Please upload a valid lung CT scan image only"
            }), 400

        file.seek(0)

        # Preprocess
        img_array = prepare_image(file)

        # Predict
        prediction = model.predict(img_array)[0]

        class_idx = np.argmax(prediction)
        confidence = float(prediction[class_idx])

        if confidence < 0.6:
            result = "Unknown / Low confidence"
        else:
            results = [
                "Lung Cancer Type A (Adenocarcinoma) - Requires medical attention",
                "Lung Cancer Type B (Large Cell Carcinoma) - Consult a doctor urgently",
                "Healthy Lungs - No signs of cancer detected",
                "Lung Cancer Type C (Squamous Cell Carcinoma) - Immediate medical consultation recommended"
            ]

            result = results[class_idx]

        return jsonify({
            "predicted_class": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)