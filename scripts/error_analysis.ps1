param(
    [string]$Model = "strong",
    [int]$NumSamples = 24
)
python -m src.error_analysis --config configs/default.yaml --model $Model --num_samples $NumSamples
