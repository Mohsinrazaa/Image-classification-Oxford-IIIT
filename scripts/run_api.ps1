$env:CHECKPOINT_PATH = "artifacts/models/strong_best.pt"
uvicorn src.api:app --host 0.0.0.0 --port 8000
