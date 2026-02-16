param(
    [string]$Model = "strong"
)
python -m src.evaluate --config configs/default.yaml --model $Model
