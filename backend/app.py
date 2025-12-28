from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from inference import predict

app = Flask(__name__)
CORS(app)

# Health check (VERY IMPORTANT FOR RENDER)
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Backend running"}), 200

@app.route("/predict", methods=["POST"])
def predict_ct():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        volume = np.load(file)
    except Exception as e:
        return jsonify({"error": "Invalid .npy file"}), 400

    prob = predict(volume)

    return jsonify({
        "cancer_probability": round(float(prob), 4),
        "prediction": "High Risk" if prob > 0.5 else "Low Risk"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
