from flask import Flask, request, jsonify
from flask_cors import CORS   # ADD THIS
import numpy as np
from inference import predict

app = Flask(__name__)
CORS(app)   # ADD THIS LINE

@app.route("/predict", methods=["POST"])
def predict_ct():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    volume = np.load(file)

    prob = predict(volume)

    return jsonify({
        "cancer_probability": round(prob, 4),
        "prediction": "High Risk" if prob > 0.5 else "Low Risk"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

