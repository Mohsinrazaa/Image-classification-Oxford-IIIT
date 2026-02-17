import { useMemo, useState } from "react";
import { predictImage } from "./api";
import type { PredictResponse } from "./types";

const DEFAULT_API = "http://127.0.0.1:8000";

export default function App() {
  const [apiBaseUrl, _] = useState(DEFAULT_API);
  const [file, setFile] = useState<File | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [predictError, setPredictError] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);


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
