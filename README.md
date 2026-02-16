# AI / Machine Learning Engineer Technical Assessment (Computer Vision)

This repository implements an end-to-end computer vision prototype for **image classification** using the **Oxford-IIIT Pet** dataset.

## 1) Problem Framing

- **Task selected:** Image classification
- **Dataset selected:** Oxford-IIIT Pet (37 pet breed classes)
- **Why this dataset:** Real-world variation in lighting, pose, background clutter, and inter-class similarity; non-trivial class count; appropriate for transfer learning and error analysis under limited compute.

## 2) Project Structure

`src/`
- `data.py`: dataset loading, preprocessing, augmentation, train/val/test split, class-weight logic
- `models.py`: baseline CNN and stronger transfer-learning model (ResNet18)
- `train.py`: training loop, early stopping, checkpointing, training metrics
- `evaluate.py`: test metrics, confusion matrix export
- `error_analysis.py`: misclassification visualization
- `inference.py`: reusable inference engine
- `infer_cli.py`: command-line inference script
- `api.py`: FastAPI inference endpoint

`configs/`
- `default.yaml`: configurable parameters for data, model, and training

`scripts/`
- PowerShell helpers for train/evaluate/inference API flow

`artifacts/`
- saved checkpoints, metrics, and plots

## 3) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4) Data Understanding and Preparation

Implemented in `src/data.py`:

- Dataset size and class metadata are loaded from Oxford-IIIT Pet.
- Label format uses class IDs mapped to class names.
- Class distribution from training split is computed and stored.
- Preprocessing:
  - resize to configurable square resolution (`224x224` by default)
  - normalization using ImageNet statistics
- Augmentation (train only):
  - random horizontal flip
  - random rotation
  - color jitter
- Class imbalance handling:
  - inverse-frequency class weights for cross-entropy loss (configurable)

## 5) Model Selection and Training

Two models are trained:

1. **Baseline model**: lightweight custom CNN (`BaselineCNN`)
2. **Stronger model**: fine-tuned `ResNet18` pre-trained on ImageNet

Design choices:
- Baseline gives a clean benchmark with low complexity.
- Transfer learning is justified due to moderate dataset size and limited compute.
- Early stopping is used to prevent overfitting and reduce unnecessary epochs.
- Hyperparameter tuning is performed through config updates (learning rate, epochs, weight decay, freeze strategy, batch size).

Run training:

```bash
python -m src.train --config configs/default.yaml --model baseline
python -m src.train --config configs/default.yaml --model strong
```

## 6) Evaluation and Error Analysis

Test evaluation includes:

- Accuracy
- Macro-F1 score
- Per-class precision/recall/F1 report
- Confusion matrix visualization

Run:

```bash
python -m src.evaluate --config configs/default.yaml --model strong
python -m src.error_analysis --config configs/default.yaml --model strong --num_samples 24
```

Generated outputs:
- `artifacts/metrics/*_test_metrics.json`
- `artifacts/plots/*_confusion_matrix.png`
- `artifacts/plots/*_misclassified_samples.png`

## 7) Inference and Deployment Readiness

### CLI inference

```bash
python -m src.infer_cli --checkpoint artifacts/models/strong_best.pt --image_path path/to/image.jpg --top_k 3
```

### REST API (FastAPI)

```bash
$env:CHECKPOINT_PATH = "artifacts/models/strong_best.pt"
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict` (multipart image upload)

Inference pipeline explicitly separates:
- input handling
- preprocessing
- model loading
- prediction formatting

## 8) Reproducibility and Engineering Quality

- Config-driven parameters (`configs/default.yaml`)
- Deterministic seed control
- Separate modules for data/training/evaluation/inference
- Logging and structured metric artifacts
- Checkpoint serialization with class names and model metadata

## 9) Compute Awareness and Tradeoffs

- Pipeline is designed for CPU, laptop GPU, or free-tier compute.
- ResNet18 chosen as a balanced accuracy/efficiency option.
- Early stopping reduces compute cost.
- Batch size and image size are configurable for memory constraints.

## 10) If Additional Compute Were Available

- Test larger transfer models (EfficientNet-B3/B4, ConvNeXt-T/S)
- Apply stronger augmentation strategies and longer fine-tuning schedules
- Run systematic hyperparameter search (Optuna or Bayesian optimization)
- Add cross-validation and ensembling
- Improve data quality via targeted relabeling of frequent failure clusters
