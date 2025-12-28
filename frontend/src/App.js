import React, { useState } from "react";
import "./App.css";

const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:5000/predict";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    if (!file) {
      setError("Please upload a .npy CT volume file.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(BACKEND_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend returned an error");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Prediction failed. Backend unreachable.");
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Lung Cancer Detection</h1>
      <p className="subtitle">Upload a preprocessed CT volume (.npy)</p>

      <input
        type="file"
        accept=".npy"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Analyzing CT..." : "Predict"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result-card">
          <h3>Prediction Result</h3>
          <p>
            <b>Risk Level:</b> {result.prediction}
          </p>
          <p>
            <b>Cancer Probability:</b> {result.cancer_probability}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
