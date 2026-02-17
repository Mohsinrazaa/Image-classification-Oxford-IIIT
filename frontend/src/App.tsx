import { useMemo, useState } from "react";
import { checkHealth, predictImage } from "./api";
import type { HealthResponse, PredictResponse } from "./types";

const DEFAULT_API = "http://127.0.0.1:8000";

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [predictError, setPredictError] = useState("");
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  async function onHealthCheck() {
    setHealthError("");
    setHealth(null);
    setIsCheckingHealth(true);
    try {
      const data = await checkHealth(apiBaseUrl);
      setHealth(data);
    } catch (error) {
      setHealthError(error instanceof Error ? error.message : "Unknown health check error");
    } finally {
      setIsCheckingHealth(false);
    }
  }

  async function onPredict() {
    if (!file) {
      setPredictError("Select an image first.");
      return;
    }

    setPredictError("");
    setPredictResult(null);
    setIsPredicting(true);
    try {
      const data = await predictImage(apiBaseUrl, file);
      setPredictResult(data);
    } catch (error) {
      setPredictError(error instanceof Error ? error.message : "Unknown prediction error");
    } finally {
      setIsPredicting(false);
    }
  }

  return (
    <main className="container">
      <h1>Computer Vision Assessment</h1>
      <p className="subtext"><b>Upload an image and get the top 3 predictions for the image.</b></p>
      <p className="subtext"><u>The model is trained on the Oxford-IIIT Pet dataset.</u></p>
      <p className="subtext"><u>Upload only images of pets.</u></p>


      {/* <section className="card">
        <h2>API Settings</h2>
        <label htmlFor="api-url">Base URL</label>
        <input
          id="api-url"
          type="text"
          value={apiBaseUrl}
          onChange={(event) => setApiBaseUrl(event.target.value.trim())}
          placeholder="http://127.0.0.1:8000"
        />
        <button onClick={onHealthCheck} disabled={isCheckingHealth}>
          {isCheckingHealth ? "Checking..." : "Check /health"}
        </button>
        {health && (
          <div className="success">
            <div>Status: {health.status}</div>
            <div>Model Loaded: {health.model_loaded ? "Yes" : "No"}</div>
          </div>
        )}
        {healthError && <div className="error">{healthError}</div>}
      </section> */}

      <section className="card">
        <h2>Prediction</h2>
        <input
          type="file"
          accept="image/*"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />

        {previewUrl && (
          <div className="preview-wrap">
            <img src={previewUrl} alt="Selected preview" className="preview" />
          </div>
        )}

        <button onClick={onPredict} disabled={isPredicting || !file}>
          {isPredicting ? "Predicting..." : "Run /predict"}
        </button>

        {predictError && <div className="error">{predictError}</div>}

        {predictResult && (
          <div className="results">
            <h3>Top Predictions</h3>
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Class Name</th>
                  <th>Class ID</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {predictResult.predictions.map((item, index) => (
                  <tr key={`${item.class_id}-${index}`}>
                    <td>{index + 1}</td>
                    <td>{item.class_name}</td>
                    <td>{item.class_id}</td>
                    <td>{(item.confidence * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
